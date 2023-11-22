import cv2
import os
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

capture = cv2.VideoCapture(0)

def detect_face():
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Faces Detected', frame)

def detect_eyes():
    # We have to detect the face and set ROI
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        # Uncomment if want to show the face frame also
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Region of Interest (ROI) for eyes in the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the ROI
        eyes = eyeCascade.detectMultiScale(roi_gray)

        # Loop over each detected eye
        for (ex, ey, ew, eh) in eyes:
            # Show rectangle around the eyes
            # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Show circle around the eyes
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            radius = int(round((ew + eh) * 0.25))
            cv2.circle(frame, eye_center, radius, (0, 0, 255), 4)
    # Display the frame with rectangles
    cv2.imshow('Eyes Detected', frame)

while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()

    # cv2.imshow('Original Webcam', frame)
    # detect_face()
    detect_eyes()
    if cv2.waitKey(1) &0XFF == ord('q'):
        break
        
# Release the capture
capture.release()
cv2.destroyAllWindows()
