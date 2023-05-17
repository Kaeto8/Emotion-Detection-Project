import cv2
from deepface import DeepFace
import numpy as np
from keras.models import load_model, model_from_json
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

#load model
#model = model_from_json(open(r'D:\nwank\Documents\Emotion Detection Project\fer.json').read())
#load weights
#model.load_weights(r'D:\nwank\Documents\Emotion Detection Project\fer.h5')

def rescaleFrame(frame, scale=1.47, scale2=2.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)


face_cascade = cv2.CascadeClassifier(r'D:\nwank\Documents\Emotion Detection Project\haarcascade_frontalface_default.xml')

#emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

#def face_detector(img):
#    #Convert image to grayscale
#    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#    faces = face_classifier.detectMultiScale(gray,1.3,5)
#    if faces is ():
#        return (0,0,0,0), np.zeros((48,48),np.uint8),img
#    
#    for (x,y,w,h) in faces:
#        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#        roi_gray = gray[y:y+h,x:x+w]
#
#    try:
#        roi_gray = cv.resize(roi_gray, (48,48), interpolation = cv.INTER_AREA)
#    except:
#        return (x,w,y,h), np.zeros((48,48), np.uint8), img
#    return (x,w,y,h), roi_gray, img

cap = cv2.VideoCapture(0)



while True:
    ret,frame = cap.read()
    #rect, face, image = face_detector(frame)
    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)

    #if np.sum([face]) != 0.0:
    #    roi = face.astype("float")/255.0
    #    roi = img_to_array(roi)
    #    roi = np.expand_dims(roi, axis=0)

    #    #make a prediction on the ROI, then lookup the class
    #    emotion = result['dominant_emotion']
    #    #preds = model.predict(roi)[0]
    #    label = str(emotion)
    #    label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
    #    cv.putText(image, label, label_position , cv.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    #else:
    #    cv.putText(image, "No Face Found", (20, 60) , cv.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,4) 

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    
    emotion = result['dominant_emotion']
    txt = str(emotion)

    cv2.putText(frame,txt,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    frame_resized = rescaleFrame(frame)
    cv2.imshow('Emotion Recognition Software',frame_resized)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()