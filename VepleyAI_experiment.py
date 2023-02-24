
import mediapipe as mp
import cv2
import time
import numpy as np
import win32api, win32con


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
#confiden = 0.4
hands = mpHands.Hands(min_detection_confidence=0.3,
               min_tracking_confidence=0.3)
mpDraw = mp.solutions.drawing_utils
lastMousePos = (0,0)

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
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            #draw a cirle on landmark 8
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    MoveMouse(((w-cx)*2,cy*2))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
