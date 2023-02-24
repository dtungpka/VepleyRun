
import mediapipe as mp
import cv2
import time
import numpy as np
import win32api, win32con
import DNN
import pickle
from tkinter import  filedialog
from VepleyAI_train import PARSE_LANDMARKS_JOINTS,Actions,transform_joints,processed_folder,VepleyAiTrain
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
#confiden = 0.4
hands = mpHands.Hands(min_detection_confidence=0.3,
               min_tracking_confidence=0.3)
mpDraw = mp.solutions.drawing_utils
lastMousePos = (0,0)
filename = filedialog.askopenfilename(initialdir = processed_folder,title = "Select file",filetypes = (("bin files","*.bin"),("all files","*.*")))
model = pickle.load(open(filename,'rb'))['model']

def MoveMouse(pos):
    #smooth the mouse movement from lastMousePos to pos
    global lastMousePos
    x,y = pos
    lastX,lastY = lastMousePos
    for i in range(1,10):
        win32api.SetCursorPos((int(lastX+i*(x-lastX)/10),int(lastY+i*(y-lastY)/10)))
        time.sleep(0.01)
        lastMousePos = pos
        

while True:
    success, img = cap.read()

    #convert the image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #process the image
    results = hands.process(imgRGB)
    lms = [[],[],[],[],[]]
    #print(results.multi_hand_landmarks)
    pos = [0,0]
    if results.multi_hand_landmarks:
        for h_,handLms in enumerate(results.multi_hand_landmarks):
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            #draw a cirle on landmark 8
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lms[h_].append([id,cx,cy])
                print(id, cx, cy)
                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    pos[0] = (w-cx)*2
                    pos[1] = cy*2
                    
            X = transform_joints(lms[h_],h_)
            X = np.array(X).reshape(1,-1)
            y_pred = DNN.predict(model,X)
            predicted = Actions[np.argmax(y_pred)]
            if predicted =='Aim':
                MoveMouse(pos)
            print(y_pred)
            #write on img
            cv2.putText(img, predicted, (10 + h_*30, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
