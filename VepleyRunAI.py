#create a udp client and send HI to localhost port 3939
import socket
import time
import sys
import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
import threading
import random
from VepleyAI_train import PARSE_LANDMARKS_JOINTS,Actions,VepleyAiTrain
from VepleyAI_acquire import WIDTH,HEIGHT,handDetector
calculate_angle = VepleyAiTrain.calculate_angle
UDP_IP  = "127.0.0.1"
UDP_PORT = 3939
'''
In the message dict:

action: head action for moving
state1:left hand state
state2: right hand state
index1_x: x coordinate of index finger of left hand
index1_y: y coordinate of index finger of left hand
index2_x: x coordinate of index finger of right hand
index2_y: y coordinate of index finger of right hand

'''
message = {'action':0,'state1':0,'state2':0,'index1_x':0, 'index1_y':0, 'index2_x':0, 'index2_y':0}
PROCESSED_FOLDER = './Processed'
Accepted_format = ['pickle','bin']
def get_file_list(path, format):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(format):
                file_list.append(file)
    return file_list



class UDP:
    def __init__(self,ip,port) -> None:
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def encode_message(self,message):
        # Convert each integer to a 4-byte byte string in network byte order
        byte_strings = [num.to_bytes(4, byteorder='big') for num in message]
        # Concatenate the byte strings to form the encoded message
        encoded_message = b''.join(byte_strings)
        #print("encoded message: %s" % encoded_message)
        return encoded_message
    def send(self,msg):
        self.socket.sendto(self.encode_message(msg),(self.ip,self.port))
    def receive(self,buffer_size):
        data, addr = self.socket.recvfrom(buffer_size)
        print("received message: %s" % data,'\nfrom',addr)
        return data
        
    def close(self):
        self.socket.close()

UDP_c = UDP(UDP_IP,UDP_PORT)
s = 0

print(s)

command_list = {' ': 0, '.': 1, '<': 2, '>':3} #reserve for cover command

while True:
    cmd = input("Enter command: ")
    if cmd not in command_list.keys():
        print("Invalid command")
        continue
    message['action'] = command_list[cmd]
    UDP_c.send(message.values())
    

#test the UDP class by set the message to random number
class main:
    def __init__(self) -> None:
        global HEIGHT, WIDTH
        self.parameters = None
        self.read_parameters()
        self.UDP = UDP(UDP_IP,UDP_PORT)
        self.detector = handDetector()
        self.joint_data = None
        #print the height, width of the vid
        
        HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.run()
        
    def read_parameters(self):
        #check if path exist
        if not os.path.exists(PROCESSED_FOLDER):
            raise Exception("Path does not exist:" + PROCESSED_FOLDER)
        #check if there is any file in the path
        file_list = get_file_list(PROCESSED_FOLDER,Accepted_format)
        if len(file_list) == 0:
            raise Exception("No file in the path")
        if len(file_list) == 1:
            self.parameters = pickle.load(open(os.path.join(PROCESSED_FOLDER,file_list[0]),"rb"))
            return
        print("Choose a file to load:")
        choice = -1
        for i in range(len(file_list)):
            print(str(i) + ":" + file_list[i])
        while choice < 0 or choice >= len(file_list):
            try:
                choice = int(input("Enter your choice:"))
            except:
                print("Invalid input")
        self.parameters = pickle.load(open(os.path.join(PROCESSED_FOLDER,file_list[choice]),"rb"))
        print("Parameters loaded")
    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            
            img_drawed = img.copy()
            img_drawed = self.detector.findHands(img_drawed)
            no_of_hand = self.detector.get_number_of_hands()
            for i in range(no_of_hand):
                lmList = self.detector.findPosition(img_drawed,i)
                #   TBA: check if the hand is in the frame
            
    
#format the message as byte[] to send to C# 
#bytesToSend = str.encode(MESSAGE)
#buffer_size = 1024

#UDPClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#UDPClient.sendto(bytesToSend, (UDP_IP, UDP_PORT))
#print("Message sent to server")
#received = UDPClient.recvfrom(buffer_size)
#print("Received message: ", received[0].decode())


#class mpHands:
#    import mediapipe as mp
#    def __init__(self,maxHands=2,tol1=.5,tol2=.5):
#        self.hands=self.mp.solutions.hands.Hands(False,maxHands,tol1,tol2)
#    def Marks(self,frame):
#        myHands=[]
#        handsType=[]
#        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#        results=self.hands.process(frameRGB)
#        if results.multi_hand_landmarks != None:
#            #print(results.multi_handedness)
#            for hand in results.multi_handedness:
#                #print(hand)
#                #print(hand.classification)
#                #print(hand.classification[0])
#                handType=hand.classification[0].label
#                handsType.append(handType)
#            for handLandMarks in results.multi_hand_landmarks:
#                myHand=[]
#                for landMark in handLandMarks.landmark:
#                    myHand.append((int(landMark.x*width),int(landMark.y*height)))
#                myHands.append(myHand)
#        return myHands,handsType