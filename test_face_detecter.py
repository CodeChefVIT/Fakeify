import cv2
# import numpy as np


face_1 = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
face_2 = cv2.CascadeClassifier('data/haarcascade_profileface.xml')

cap = cv2.VideoCapture(0)

while 1:
    _, frame = cap.read()
    faces_1 = face_1.detectMultiScale(frame, 1.3, 5)
    faces_2 = face_2.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces_1:
        cv2.rectangle(frame, (x, y), (x + w + 5, y + h + 5), (678, 345, 12), 2)
        roi_gray = frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    for (x, y, w, h) in faces_2:
        cv2.rectangle(frame, (x, y), (x + w + 5, y + h + 5), (0, 345, 12), 2)
        roi_gray = frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()

