from flask import Flask, render_template, Response
import cv2
import os
import cvzone
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)
# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# # cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    # cap= cv2.VideoCapture("Resources/Videos/4.mp4")
    # cap= cv2.VideoCapture(0)

    detector = PoseDetector()
    shirtsFolderPath = "Resources/Shirts"
    listShirts = os.listdir(shirtsFolderPath)
    fixedRatio = 262 / 190
    shirtRatioHeightWidth = 581 / 440
    imageNumber = 0
    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            lm11 = lmList[11][0:2]
            lm12 = lmList[12][0:2]
            imgShirt = cv2.imread(os.path.join(shirtsFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            imgShirt = cv2.resize(imgShirt, (int(widthOfShirt), int(widthOfShirt * shirtRatioHeightWidth)))
            # resized_image = cv2.resize(imgShirt, (1400, 750))

            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * currentScale), int(48 * currentScale)
            try:
                img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except:
                pass
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
