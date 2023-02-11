from argparse import Action
from asyncore import read
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import mediapipe
import os
import time
import pickle
import sys
from DNN import NeuralNetwork

TRAIN_TEST_RATIO = 0.8
#H: 480
#W: 640
H = 480
W = 640
dataFiles = "Datasets"
processed_folder = "./Processed/"
Excluded = []
parent_folder = True
Actions = ['Idle',
           'Pickup_item',
           'Use_item',
           'Aim',
           'Shoot'
           ]
IGNORE_RIGHT_HAND = False
FLIP_RIGHT_HAND = True

#train the neural net with 256 
LAYERS = [1, 256, 256, 256, len(Actions)]

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
        #self.read_data(filesPath , parentFolder, excluded)
        pass
    def read_data(self,path: str,_parentFolder: bool = parent_folder, _excluded: list = Excluded):
        if not os.path.exists(path):
            raise Exception("Path does not exist:" + path)
        #check if detail.json is in path
        if os.path.exists(os.path.join(path,"detail.json")):
            #read the 'total' key in file and save to NO_OF_SAMPLES
            global NO_OF_SAMPLES,Actions,W,H
            with open(os.path.join(path,"detail.json"),"r") as f:
                detail = json.load(f)
                NO_OF_SAMPLES = detail['total']
                Actions = detail.keys()
                W = detail['W']
                H = detail['H']
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
        pickle.dump(self.data,open(processed_folder+dataFiles+".pickle","wb"))
        print("Data processed,saved to "+processed_folder+dataFiles+".pickle")
        
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
                        self.data['ID'][int(ID)] = int(ID+hand)
                        self.data['Y'][int(ID)] = j
                        landmarks = {point[0]:np.array(point[1:]) for point in raw_datas[ID][hand]}
                        if FLIP_RIGHT_HAND and hand == '1':
                            for k in landmarks:
                                landmarks[k][0] = W - landmarks[k][0]
                        for i,(_p1,_p2) in enumerate(PARSE_LANDMARKS_JOINTS):
                            if _p1 in landmarks and _p2 in landmarks:
                                self.data['X'][i][int(ID)] = self._calculate_angle(landmarks[_p1],landmarks[_p2])
                            else:
                                self.data['X'][i][int(ID)] = 0
            print("\rProcessed "+action,f'{j+1} out of {len(Actions)} ',end='\n')                         
    def _calculate_angle(self,landmark1, landmark2):
        return np.math.atan2(np.linalg.det([landmark1, landmark2]), np.dot(landmark1, landmark2))
    def load(self):
        if self.data == None:
            files = os.listdir(processed_folder)
            if len(files) == 0:
                print("No processed data found, please process data first")
                return
            else:
                #read the lastest data
                self.data = pickle.load(open(processed_folder+files[-1],"rb"))
                print("Loaded data from "+processed_folder+files[-1])
        
        #spilt the data
        self._split_data()
    def _split_data(self):
        global TRAIN_TEST_RATIO,NO_OF_SAMPLES
        train_size = int(NO_OF_SAMPLES*TRAIN_TEST_RATIO)
        self.X = self.data['X'][:train_size,:]
        self.Y = self.data['Y'][:train_size]
        self.X_test = self.data['X'][train_size:,:]
        self.Y_test = self.data['Y'][train_size:]
        #print all the shape
        print("X shape: ",self.X.shape)
        print("Y shape: ",self.Y.shape)
        print("X_test shape: ",self.X_test.shape)
        print("Y_test shape: ",self.Y_test.shape)
    def train(self):
        self.num_feature = self.X.shape[1]
        self.num_class = len(Actions)
        self.model =  NeuralNetwork(num_features=self.num_feature, num_classes=self.num_class)
        self.model.fit(self.X,self.Y)
        
        
        
        
#check for paramater, if --tune then run class
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
        instance.read_data(dataFiles,not s, Excluded)
        instance.save_processed()
        instance.load()
        instance.train()
      
    
    