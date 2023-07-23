import cv2 as cv
import numpy as np


video = cv.VideoCapture("AI BOOTCAMP PROJECT/mixkit-traveling-on-the-highway-on-a-sunny-day-42368-medium.mp4")

objectDetector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    var1, frame = video.read()


    #Generate ROI

    roi = frame[360:720, 0:1280]

    #object detection
    mask = objectDetector.apply(roi)
    var3, mask = cv.threshold(mask, 230, 255, cv.THRESH_BINARY)
    contours, var2 = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        #remove small contours
        area = cv.contourArea(c)
        if area > 600:
            #cv.drawContours(roi, c, -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(roi, (x, y), (x +w, y + h), (0, 255, 0), 2)

            detections.append([x, y, w, h])
            print(detections)

            cv.putText(frame, 
                       "COLLISION WARNING", 
                       (150, 150), 
                       cv.FONT_HERSHEY_SIMPLEX, 1.3, 
                       (0, 0, 255), 
                       2, 
                       cv.LINE_4)

    
    cv.imshow("roi", roi)
    cv.imshow("Frame", frame)
    cv.imshow("Mask", mask)

    key = cv.waitKey(1) #It waits 1 milisecond, before moving to next frame

    if key == 27: #key 27 is s
        break

frame.release()
cv.destroyAllWindows()
