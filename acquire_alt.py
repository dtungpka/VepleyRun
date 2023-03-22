#using mediapipe to detect hand

from argparse import Action
import cv2
import mediapipe as mp
import time
import numpy as np
import os
import json
import shutil
from console_progressbar import ProgressBar
import threading

guide_path = './Guides/'
Actions = ['Idle',
           'Pickup_item',
           'Use_item',
           'Aim',
           'Shoot'
           ]

NUMBER_OF_SAMPLES = 1000
CHANGE_POSE_EVERY = 0.2
PAUSE_TIME = 2.0
CAPTURE_RATE = 1 # capture every n frame
Path = ""
lock = None
HEIGHT = 480
WIDTH = 640
remaining_time = 0


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=mode,
    max_num_hands=self.maxHands,
    min_detection_confidence=self.detectionCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        #print(self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
           
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmList
    def get_bouding_box(self,img,handNo=0):
        if self.results.multi_hand_landmarks:
            lmList = self.findPosition(img, handNo=handNo, draw=False)
            x_min = min(lmList, key=lambda x: x[1])[1] 
            x_max = max(lmList, key=lambda x: x[1])[1] 
            y_min = min(lmList, key=lambda x: x[2])[2]
            y_max = max(lmList, key=lambda x: x[2])[2] 
            #measure the ratio between x_min, x_max, y_min, y_max and the image size, ajust the bounding box to larger according to the ratio
            offset = 0.07
            x_min = int(x_min - (x_max - x_min)*offset)
            x_max = int(x_max + (x_max - x_min)*offset)
            y_min = int(y_min - (y_max - y_min)*offset)
            y_max = int(y_max + (y_max - y_min)*offset)
            if x_min < 0:
                x_min = 0
            if x_max > WIDTH:
                x_max = WIDTH
            if y_min < 0:
                y_min = 0
            if y_max > HEIGHT:
                y_max = HEIGHT

                

            return x_min, x_max, y_min, y_max
        return 0,0,0,0
    def draw_bouding_box(self,img,handNo=0):
        x_min, x_max, y_min, y_max = self.get_bouding_box(img, handNo=handNo)
        cv2.rectangle(img, (x_min , y_min), (x_max , y_max ), (0, 255, 0), 2)
        return img

    def get_number_of_hands(self):
        if self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0
    ID = 0
    capturing = False
    thr_running = True
    waitID = -1
def checkpoint():
    global remaining_time, PAUSE_TIME
    while handDetector.thr_running:
        time.sleep(0.1)
        
        if handDetector.waitID == handDetector.ID or lock.locked():
            continue
        if handDetector.ID % NUMBER_OF_SAMPLES == 0 and not handDetector.capturing:
            input('\rReady your pose and press ENTER to continue')
            remaining_time = PAUSE_TIME
            while remaining_time > 0:
                print(f"\rContinue in {round(remaining_time,2)}",end='')
                time.sleep(0.1)
                remaining_time -= 0.1
            handDetector.capturing = True
            handDetector.waitID = handDetector.ID
            continue
        if handDetector.ID % round(NUMBER_OF_SAMPLES* CHANGE_POSE_EVERY) == 0:
            handDetector.capturing = False
            #input('\rPlease change your pose for better data and press ENTER to continue')
           
            remaining_time = PAUSE_TIME
            while remaining_time > 0:
                print(f"\rContinue in {round(remaining_time,2)}",end='')
                time.sleep(0.1)
                remaining_time -= 0.1
            handDetector.capturing = True
            handDetector.waitID = handDetector.ID
            


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    #print the height, width of the vid
    global HEIGHT, WIDTH
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height: ", HEIGHT)
    print("Width: ", WIDTH)
    detector = handDetector()
    captured_action = {}
    current_action = 0;
    print("\nStart sampling..")
    print(f"Sampling {Actions[current_action]}\n")
    fr_count = 0
    cap_thread = threading.Thread(target=checkpoint)
    cap_thread.start()
    d_guide = ''
    with open(guide_path + 'detail_guide.txt', 'r') as f:
        d_guide = f.read()
    print(d_guide.replace('[pose]',Actions[current_action]))
    pb = ProgressBar(total=NUMBER_OF_SAMPLES, suffix=Actions[current_action], decimals=3, length=50, fill='=', zfill=' ')
    global remaining_time
    while current_action < len(Actions):
        
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img_drawed = img.copy()
        img_drawed = detector.findHands(img_drawed,draw=False)
        
        
        no_of_hand = detector.get_number_of_hands()
        lock.acquire()
        for i in range(no_of_hand):
            bouding_box = detector.get_bouding_box(img_drawed,i)
            img_drawed = detector.draw_bouding_box(img_drawed,i)
            lmList = detector.findPosition(img_drawed,i,draw=False)
            if fr_count % CAPTURE_RATE == 0 and handDetector.capturing:
                if len(lmList) != 0:
                    if i == 0:
                        d = {'landmarks':lmList, 'bouding_box':bouding_box}
                        captured_action[handDetector.ID] = {i:d}   
                    else:
                        d = {'landmarks':lmList, 'bouding_box':bouding_box}
                        captured_action[handDetector.ID][i] = d
                if i == no_of_hand-1:
                    handDetector.ID += 1
                    pb.print_progress_bar(handDetector.ID % NUMBER_OF_SAMPLES)
        if handDetector.ID % NUMBER_OF_SAMPLES == 0 and handDetector.capturing and (handDetector.waitID % NUMBER_OF_SAMPLES != 0):
            #save captured_action to json and clear content of var
            with open(os.path.join(Path, f'{Actions[current_action]}.json'), 'w') as outfile:
                json.dump(captured_action, outfile)
                captured_action = {}
                outfile.flush()
            handDetector.capturing = False
            if current_action == len(Actions) - 1:
                handDetector.thr_running = False
                lock.release()
                
                return
            current_action += 1
            print(f"Sampling {Actions[current_action]}\n")
            print(d_guide.replace('[pose]',Actions[current_action]))
            pb = ProgressBar(total=NUMBER_OF_SAMPLES, suffix=Actions[current_action], decimals=3, length=50, fill='=', zfill=' ')
        lock.release()
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        fr_count += 1
        
        cv2.putText(img_drawed, "FPS: "+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 3)
        if handDetector.capturing:
            cv2.putText(img_drawed,f"Sampling {Actions[current_action]}", (60, 450), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 100, 255), 3)
        else:
            cv2.putText(img_drawed,f"Waiting to sample {Actions[current_action]}", (60, 450), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 100, 255), 3)
        cv2.putText(img_drawed,f"ID: {handDetector.ID}", (10, 450), cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255), 3)
        #if remaining_time > 0 then put text
        if remaining_time > 0:
            cv2.putText(img_drawed,f"Continue in {round(remaining_time,2)}", (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,(255, 255, 255), 3)
        #save the img
        #flip the img horizontal
        cv2.imshow("Frame", img_drawed)
        if handDetector.capturing:
            cv2.imwrite(os.path.join(Path, Actions[current_action], f"{handDetector.ID}.jpg"), img)
            cv2.imwrite(os.path.join(Path, Actions[current_action], f"{handDetector.ID}_drawed.jpg"), img_drawed)
        cv2.waitKey(1)
        
if __name__ == "__main__":
    lock = threading.Lock()
    Path = input("Enter Dataset name: ")
    for action in Actions:
        if not os.path.exists(os.path.join(Path, action)):
            os.makedirs(os.path.join(Path, action))
    with open(guide_path+'guide.txt','r') as f:
        print(f.read())
        input('\nEnter to continue..')
    main()
    handDetector.thr_running = False
    #get the len of all file in folder with same name as Actions[current_action]
    detail_f = open(os.path.join(Path, "detail.json"), "w")
    #write a dict to detail_f, contain all action label and number of samples
    detail = {}
    f_sum = 0
    for action in Actions:
        detail[action] = len(os.listdir(os.path.join(Path, action)))
        f_sum += detail[action]
    detail["total"] = f_sum
    detail["height"] = HEIGHT
    detail["width"] = WIDTH
    json.dump(detail, detail_f)
    detail_f.close()
    print("\r\nThank you for your time!")
    #add to zip
    shutil.make_archive(f'VepleyAI_dataset_{Path}', 'zip', Path)
    os.rename(f'VepleyAI_dataset_{Path}.zip', f'VepleyAI_dataset_{Path}.VAID')
    os.startfile(os.path.realpath(f'VepleyAI_dataset_{Path}.VAID'))
    #remove folder Path and all item inside
    shutil.rmtree(Path)
        
    input("Press ENTER to exit")
