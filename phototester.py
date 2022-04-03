import cv2
import videotester
import sys
import torchvision.transforms as transforms
import resnet9
import torch

model=resnet9.ResNet9(1, 7)
model.load_state_dict(torch.load("fer2013-resnet9.pth",map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])


classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def detect_emotion(gray_img):
    inp_img = cv2.resize(gray_img, (48, 48))
    inp_img=transform(inp_img)
    inp_img=inp_img.reshape(1,1,48,48)
    predictions = model(inp_img)

        # find max indexed array
    _, preds  = torch.max(predictions, dim=1)

        
    emotion = classes[preds[0].item()]

    return emotion


if __name__ == "__main__":
    test_image=sys.argv[1]
    img=cv2.imread(test_image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img=cv2.imshow('image',img)
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    
    if faces_detected != ():
        
        x,y,w,h=faces_detected[0]
        #print(x,y,w,h)
        emotion=detect_emotion(gray_img[x:x+w,y:y+h])
        image=cv2.rectangle(img, (x, y), (x + w, y + h), (255, 100, 0), thickness=5)
        image=cv2.putText(image,emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(emotion)
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Face not detected")


