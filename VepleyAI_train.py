from argparse import Action
from asyncore import read
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import random
import os
import time
import pickle
import sys
import DNN
from VepleyAI_acquire import Actions
TRAIN_TEST_RATIO = 0.8
#H: 480
#W: 640
H = 480
W = 640
dataFiles = "Datasets"
processed_folder = "./Processed/"
Excluded = []
parent_folder = True
IGNORE_RIGHT_HAND = False
FLIP_RIGHT_HAND = True
LEARNING_RATE = 0.0001
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
                                self.data['X'][i][int(ID)] = self.calculate_angle(landmarks[_p1],landmarks[_p2])
                            else:
                                self.data['X'][i][int(ID)] = 0
            print("\rProcessed "+action,f'{j+1} out of {len(Actions)} ',end='\n')     
    def calculate_angle(self,landmark1, landmark2):
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
        self.X = self.data['X'][:train_size,:].T
        self.Y = self.data['Y'][:train_size].T
        #from Y (800,) to (800,5)
        self.Y = np.eye(len(Actions))[self.Y.astype(int)].T

        self.X_test = self.data['X'][train_size:,:].T
        self.Y_test = self.data['Y'][train_size:].T
        #print all the shape
        print("X shape: ",self.X.shape)
        print("Y shape: ",self.Y.shape)
        print("X_test shape: ",self.X_test.shape)
        print("Y_test shape: ",self.Y_test.shape)
    def train(self):
        self.num_feature = self.X.shape[0]
        self.num_class = len(Actions)
        self.parameters = DNN.initialize_parameters_deep([self.num_feature, self.num_feature,  self.num_class])
        _parameters = self.parameters
        history = []
        cost = 100
        for i in range(0, 90000):
            AL, caches = DNN.L_model_forward(self.X, _parameters)
            grads = DNN.L_model_backward(AL, self.Y, caches)
            _cost = np.sum(DNN.compute_cost(AL, self.Y))
            _parameters = DNN.update_parameters(_parameters, grads, learning_rate = LEARNING_RATE)
            if (_cost < cost ):
                self.parameters = _parameters
                cost = _cost
            history.append(np.sum(_cost))
            if i % 100 == 0:
                print ("\rCost after iteration %i: " %(i),np.sum(cost),end='')
       
        plt.plot(history)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(LEARNING_RATE))
        plt.show()
        self.predict_random()
        self.save_parameters()
    def predict_random(self):
        if self.X_test is None:
            print("No data to predict")
            return
        for i in range(0,10):
            index = random.randint(0,self.X_test.shape[1])
            print("Predicted: ",Actions[DNN.predict(self.X_test[:,index],self.parameters)])
            print("Actual: ",Actions[self.Y_test[:,index].argmax()])
    def save_parameters(self):
        if input("Save parameters? (y/n)").lower() != "y":
            return
        global processed_folder
        if not os.path.exists(processed_folder):
            os.mkdir(processed_folder)
        pickle.dump(self.parameters,open(processed_folder+dataFiles+"_parameters.pickle","wb"))
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
        instance.read_data(dataFiles,not s, Excluded)
        instance.save_processed()
        instance.load()
        instance.train()
      
    
    