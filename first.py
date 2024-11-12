import os
import mediapipe as mp
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector

# cap= cv2.VideoCapture("Resources/Videos/1.mp4")
cap= cv2.VideoCapture(0)
detector = PoseDetector()

shirtsFolderPath = "Resources/Shirts"

# screen_width = 1920  # Change this to your screen width
# screen_height = 1080  # Change this to your screen height

listShirts = os.listdir(shirtsFolderPath)
print(listShirts)

fixedRatio = 262/190 ;  # widthOfShirt / WidthOfPoints 11 to 12

shirtRatioHeightWidth=581/440


imageNumber=0
imageButtonRight=cv2.imread("Resources/button.png",cv2.IMREAD_UNCHANGED)
imageButtonLeft=cv2.flip(imageButtonRight,1)

selectionSpeed=10
counterRight=0
counterLeft=0

#For fulls screen using Window fullscreen
# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
while True:
    success, img = cap.read()
    # img =lmList[11][1:3]
    img = detector.findPose(img)
    # img=cv2.flip(img,1)
    lmList,bboxInfo=detector.findPosition(img,bboxWithHands=False,draw=False)
    if lmList:
        lm11 = lmList[11][0:2]
        lm12 = lmList[12][0:2]

        imgShirt = cv2.imread(os.path.join(shirtsFolderPath,listShirts[imageNumber]),cv2.IMREAD_UNCHANGED)

        widthOfShirt=int((lm11[0]-lm12[0])*fixedRatio)
        print(widthOfShirt)
        # print(widthOfShirt)
        imgShirt=cv2.resize(imgShirt,(int(widthOfShirt),int(widthOfShirt*shirtRatioHeightWidth)))

        currentScale = ( lm11[0]-lm12[0] ) /190

        offset = int(44*currentScale),int(48*currentScale)

        try:
         img = cvzone.overlayPNG(img,imgShirt,(lm12[0]-offset[0],lm12[1]-offset[1]-10))
        except :
            pass

        # img = cvzone.overlayPNG(img,imageButtonRight,(1074,293))
        # img = cvzone.overlayPNG(img, imageButtonLeft, (72, 293))

        if lmList[16][0] < 300:
            counterRight += 1
            cv2.ellipse(img, (139, 360), (66, 66), 0, 0,
                        counterRight * selectionSpeed, (0, 255, 0), 20)
            if counterRight * selectionSpeed > 360:
                counterRight = 0
                if imageNumber < len(listShirts) - 1:
                    imageNumber += 1

        elif lmList[15][0] > 900:
            counterLeft += 1
            cv2.ellipse(img, (1138, 360), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
            if counterLeft * selectionSpeed > 360:
                counterLeft = 0
                if imageNumber > 0:
                    imageNumber -= 1

        else:
            counterRight=0
            counterLeft=0
    resized_image = cv2.resize(img, (1400, 750))
    cv2.imshow("Image", resized_image)
    cv2.waitKey(1)

