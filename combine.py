import cv2
from cv2 import VideoCapture
from deepface import DeepFace
import numpy as np
from keras.models import load_model, model_from_json
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

face_classifier = cv2.CascadeClassifier(r'D:\nwank\Documents\Emotion Detection Project\haarcascade_frontalface_default.xml')

def face_detector(img):
    #Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48),np.uint8),img

    try:
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((48,48), np.uint8), img
    return (x,w,y,h), roi_gray, img

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    if np.sum([face]) != 0.0:
        roi = face.astype("float")/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
    
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
            try:
                analyze = DeepFace.analyze(frame,actions=['emotions'])
                label = analyze['dominant emotion']
                label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
                cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
            except:
                cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    
    cv2.imshow('Emotion Detector',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()