import numpy as np
import face_recognition
import cv2
import os
import progressbar

bar = progressbar.ProgressBar(maxval=100, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
imgCount=0

faceName = input("\nHey Let me learn you..whats your name? ")
path = "faces/train/"+faceName+"/"
if not os.path.exists(path):
    os.makedirs(path)
print("\nWriting you to path ", path)
input("\nLets start , Hit Enter when you are ready ? ")
print("\nHold On, I'm learning\n")
bar.start()
while(imgCount < 95):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #print(imgCount)
        imgCount = imgCount+1
        #print(imgCount)
        mouth_roi = img[y:y + h, x:x + w]
        if imgCount%10==0 :
            res = cv2.resize(mouth_roi,(200, 200), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(path+str(imgCount)+".png", res)
            # image = face_recognition.load_image_file(path+str(imgCount)+".png")
            # face_encoding = face_recognition.face_encodings(image)
            # if(len(face_encoding)>0):
            #     known_face_encodings.append(face_encoding[0])
            #     known_face_names.append(faceName)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        bar.update(imgCount+1)

    cv2.imshow('frame',img)
    #cv2.imshow('face', mouth_roi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
bar.finish()
print("\nI know you now")
cap.release()
cv2.destroyAllWindows()