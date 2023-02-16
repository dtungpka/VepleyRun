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
from VepleyAI_train import PARSE_LANDMARKS_JOINTS,Actions,VepleyAiTrain
from VepleyAI_acquire import WIDTH,HEIGHT
calculate_angle = VepleyAiTrain.calculate_angle
UDP_IP  = "127.0.0.1"
UDP_PORT = 3939
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
        return encoded_message
    def send(self,msg):
        self.socket.sendto(self.encode_message(msg),(self.ip,self.port))
    def receive(self,buffer_size):
        data, addr = self.socket.recvfrom(buffer_size)
        return data
        
    def close(self):
        self.socket.close()






class main:
    def __init__(self) -> None:
        self.parameters = None
        self.read_parameters()
        self.UDP = UDP(UDP_IP,UDP_PORT)
        
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
    
        
    
            
    
#format the message as byte[] to send to C# 
#bytesToSend = str.encode(MESSAGE)
#buffer_size = 1024

#UDPClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#UDPClient.sendto(bytesToSend, (UDP_IP, UDP_PORT))
#print("Message sent to server")
#received = UDPClient.recvfrom(buffer_size)
#print("Received message: ", received[0].decode())
