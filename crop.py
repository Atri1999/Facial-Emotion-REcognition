import cv2
import sys

test_image=sys.argv[1]
img=cv2.imread(test_image)
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_detected = face_haar_cascade.detectMultiScale(img, 1.32, 5)
x,y,w,h=faces_detected[0]

cv2.imshow('image',img[x:x+w,y:y+h])
cv2.waitKey(0)
cv2.destroyAllWindows()