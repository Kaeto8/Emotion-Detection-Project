import cv2
from cv2 import VideoCapture
from deepface import DeepFace
import numpy as np
from keras.models import load_model, model_from_json
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

face_classifier = cv2.CascadeClassifier(r'D:\nwank\Documents\Emotion Detection Project\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation = cv2.INTER_AREA)

        try:
            analyze = DeepFace.analyze(frame,actions=['emotions'])
            label = analyze['dominant emotion']
            label_position = (x, y-10)
            cv2.putText(frame, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
        except:
            cv2.putText(frame, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()