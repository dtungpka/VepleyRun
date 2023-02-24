from argparse import Action
from asyncore import read
from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import os
import time
import pickle
import sys
import DNN
import shutil
from VepleyAI_acquire import Actions
from tkinter import filedialog

TRAIN_TEST_RATIO = 0.8
#H: 480
#W: 640
H = 480
W = 640
dataFiles = "D:\\2021-2022\\Research\\Dataset\\VepleyAI"
processed_folder = "D:\\2021-2022\\Research\\VepleyRunAI\\Processed"
Excluded = []
parent_folder = True
IGNORE_RIGHT_HAND = False
FLIP_RIGHT_HAND = True
LEARNING_RATE = 0.0001
ITERATIONS = 20
#train the neural net with 256 
LAYERS = [512, 256, 256, len(Actions)]
ACTIVATION = [ "relu", "relu", "relu", "sigmoid"]
PARSE_LANDMARKS_JOINTS = [    
    [0, 1], [1, 2], [2, 3], [3, 4], # thumb
    [0, 5],[5, 6], [6, 7], [7, 8], # index finger
    [5, 9],[9,10],[10, 11], [11, 12],# middle finger
    [9, 13],[13, 14],[14, 15],[15, 16], # ring finger
    [13, 17],  [17, 18], [18, 19], [19,20]   # little finger
]
# relu
# sigmoid



NO_OF_SAMPLES = 1000
np.random.seed(1)
class VepleyAiTrain():
    def __init__(self, filesPath = "Datasets",layers = LAYERS,train_ratio = 0.8 ,parentFolder = True,excluded = Excluded) -> None: 
        self.date = None
        self.data = None
        #self.read_data(filesPath , parentFolder, excluded)
        pass
    def read_data(self,path: str,_parentFolder: bool = parent_folder, _excluded: list = Excluded):
        if not os.path.exists(path):
            raise Exception("Path does not exist:" + path)
        #check if detail.json is in path
        self.unpack(path,_parentFolder)
        self.get_details(path,_parentFolder)
        self.data = {'X':np.zeros((len(PARSE_LANDMARKS_JOINTS),NO_OF_SAMPLES)),
                     'Y':np.zeros(NO_OF_SAMPLES),
                     'ID':np.zeros(NO_OF_SAMPLES)}
        if _parentFolder:
            for folder in os.listdir(path):
                if folder not in _excluded and os.path.isdir(os.path.join(path,folder)):
                    print("Processing dataset: "+folder)
                    self._append_data(os.path.join(path,folder))
        else:
            self._append_data(path)
        self.data['X'] = self.data['X'].T
    def save_processed(self):
        global processed_folder
        if not os.path.exists(processed_folder):
            os.mkdir(processed_folder)
        pickle.dump(self.data,open(os.path.join(processed_folder,dataFiles)+".bin","wb"))
        print("Data processed,saved to "+processed_folder+dataFiles+".bin")
    def unpack(self,path,parentFolder):
        if parentFolder:
            for VAID in os.listdir(path):
                #using shutil to unpack the zip file
                #if file end with .VAID,  unpack as .zip
                if VAID.endswith(".VAID") and not os.path.exists(os.path.join(path,VAID[:-5])):
                    print("Unpacking "+VAID)
                    shutil.unpack_archive(os.path.join(path,VAID),os.path.join(path,VAID[:-5]),"zip")
        else:
            pass
    def get_details(self,path, parentFolder):
        global NO_OF_SAMPLES,Actions,W,H
        if parentFolder:
            NO_OF_SAMPLES = 0
            for folder in os.listdir(path):
                if os.path.isdir(os.path.join(path,folder)):
                    #read detail.json, sum the total
                    with open(os.path.join(path,folder,"detail.json"),"r") as f:
                        detail = json.load(f)
                        NO_OF_SAMPLES += detail['total']*2
                        W = detail['width']
                        H = detail['height']
        else:
            with open(os.path.join(path,"detail.json"),"r") as f:
                detail = json.load(f)
                NO_OF_SAMPLES = detail['total']*2

                W = detail['width']
                H = detail['height']
        
        
    def _append_data(self,path: str):
        global IGNORE_RIGHT_HAND,FLIP_RIGHT_HAND,PARSE_LANDMARKS_JOINTS
        if not os.path.exists(path):
            raise Exception("Path does not exist:" + path)
        for j,action in enumerate(Actions):
            print("Processing "+action,f'{j+1} out of {len(Actions)} ',end='')
            with open(os.path.join(path,action+".json"),"r") as f:
                raw_datas = json.load(f)
                for ID in raw_datas:
                    for hand in raw_datas[ID]:
                        if IGNORE_RIGHT_HAND and hand == '1':
                            continue
                        if hand == '0':
                            self.data['ID'][int(ID)] = int(ID+hand)
                            self.data['Y'][int(ID)] = j
                        else:
                            self.data['ID'][int(ID)*2] = int(ID+hand)
                            self.data['Y'][int(ID)*2] = j
                        landmarks = {point[0]:np.array(point[1:]) for point in raw_datas[ID][hand]}
                        if FLIP_RIGHT_HAND and hand == '1':
                            for k in landmarks:
                                landmarks[k][0] = W - landmarks[k][0]
                        for i,(_p1,_p2) in enumerate(PARSE_LANDMARKS_JOINTS):
                            if _p1 in landmarks and _p2 in landmarks:
                                tmp = self.calculate_angle(landmarks[_p1],landmarks[_p2])
                                if hand == '0':
                                    self.data['X'][i][int(ID)] = tmp
                                self.data['X'][i][int(ID)*2] = tmp
                            else:
                                if hand == '0':
                                    self.data['X'][i][int(ID)] = 0
                                self.data['X'][i][int(ID)*2] = 0
            print("\rProcessed "+action,f'{j+1} out of {len(Actions)} ',end='\n')     
    def calculate_angle(self,landmark1, landmark2):
        return np.math.atan2(np.linalg.det([landmark1, landmark2]), np.dot(landmark1, landmark2))
    def load(self):
        if self.data == None:
            #prompt to select file, end with .bin
            global processed_folder
            filename = filedialog.askopenfilename(initialdir = processed_folder,title = "Select file",filetypes = (("bin files","*.bin"),("all files","*.*")))
            if filename:
                self.data = pickle.load(open(filename,"rb"))
                print("Data loaded from "+filename)
            
        
        #spilt the data
        self._split_data()
    def _split_data(self):
        global TRAIN_TEST_RATIO,NO_OF_SAMPLES

        pmt = np.random.permutation(NO_OF_SAMPLES)
        self.data['X'] = self.data['X'][pmt,:]
        self.data['Y'] = self.data['Y'][pmt]
        
        train_size = int(NO_OF_SAMPLES*TRAIN_TEST_RATIO)
        self.X = self.data['X'][:train_size,:]
        self.Y = self.data['Y'][:train_size]
        #from Y (800,) to (800,5)
        self.Y = np.eye(len(Actions))[self.Y.astype(int)]
        
        self.X_test = self.data['X'][train_size:,:]
        self.Y_test = np.eye(len(Actions))[(self.data['Y'][train_size:]).astype(int)]
        #print all the shape
        print("X shape: ",self.X.shape)
        print("Y shape: ",self.Y.shape)
        print("X_test shape: ",self.X_test.shape)
        print("Y_test shape: ",self.Y_test.shape)
    def train(self):
        global LAYERS, ITERATIONS, ACTIVATION
        if self.X is None:
            print("No data to train")
            return
        self.model = DNN.create_model(LAYERS,ACTIVATION)
        self.parameters = DNN.train_model(self.model,self.X,self.Y,self.X_test,self.Y_test,ITERATIONS)
        self.save_parameters()
        DNN.plot_model_history(self.parameters)
        self.predict_random()
        
    def predict_random(self):
        if self.X_test is None:
            print("No data to predict")
            return
        for i in range(0,10):
            index = random.randint(0,self.X_test.shape[1])
            X_pred = self.X_test[index,:]
            Y_pred = DNN.predict(self.model,X_pred.reshape(1,20))
            print("Predicted: ",Actions[np.argmax(Y_pred)])
            Y_true = self.Y_test[index,:]
            print("True: ",Actions[np.argmax(Y_true)])
            
    def save_parameters(self):
        if input("Save parameters? (y/n)").lower() != "y":
            return
        global processed_folder
        if not os.path.exists(processed_folder):
            os.mkdir(processed_folder)
        pickle.dump(self.parameters,open(os.path.join(processed_folder,dataFiles)+"_parameters.bin","wb"))
        print("Parameters saved to "+processed_folder+dataFiles+"_parameters.bin")
        
        
        
        
if __name__ == "__main__":
    instance = VepleyAiTrain()
    s = "--single" in sys.argv or "-s" in sys.argv
    if len(sys.argv) > 1 :
      if "--process" in sys.argv or "-p" in sys.argv:
        instance.read_data(dataFiles,not s, Excluded)
        instance.save_processed()
      else:
          instance.load()
      if '--train' in sys.argv or '-t' in sys.argv:
         instance.train()   
    else:
        print("No arguments given, please use --process or --train")
        input("Press enter for debug mode")
        #instance.read_data(dataFiles,not s, Excluded)
        #instance.save_processed()
        instance.load()
        instance.train()
      
    
    