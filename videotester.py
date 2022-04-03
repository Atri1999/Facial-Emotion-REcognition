import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
import matplotlib.pyplot as plt
import numpy as np
import torch
import resnet9


def facial_emotion_detection(faces_detected,test_img,gray_img):
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 100, 0), thickness=5)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        img_pixels=torch.from_numpy(img_pixels)
        #print(img_pixels.shape)
        
        img_pixels=img_pixels.reshape(1,1,48,48)
        predictions = model(img_pixels)

        # find max indexed array
        _, preds  = torch.max(predictions, dim=1)

        
        predicted_emotion = classes[preds[0].item()]

        cv2.putText(test_img,predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        return predicted_emotion
    


# load model
model=resnet9.ResNet9(1, 7)
model.load_state_dict(torch.load("fer2013-resnet9.pth",map_location=torch.device('cpu')))
model.eval()

#classes of emotions
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

if __name__=="__main__":
    # for detecting faces

    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        

        facial_emotion_detection(faces_detected,test_img,gray_img)

        #x,y,w,h=faces_detected
        #print(x,y,w,h)
            
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img) 


        
        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows