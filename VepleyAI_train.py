
from argparse import Action
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
VALIDATION_FILES = ['VepleyAI_dataset_Hoang1', 'VepleyAI_dataset_Nam1']


NO_OF_SAMPLES = 1000
np.random.seed(1)

def transform_joints(joints: list,hand = 0):
        X = np.zeros((len(PARSE_LANDMARKS_JOINTS),))
        landmarks = {point[0]:np.array(point[1:]) for point in joints}
        if FLIP_RIGHT_HAND and hand == 1:
            for k in landmarks:
                landmarks[k][0] = W - landmarks[k][0]
        for i,(_p1,_p2) in enumerate(PARSE_LANDMARKS_JOINTS):
            if _p1 in landmarks and _p2 in landmarks:
                tmp = VepleyAiTrain.calculate_angle(None,landmarks[_p1],landmarks[_p2])
                X[i] = tmp
            else:
                X[i] = 0
        return X
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)
def precision_score(y_true, y_pred):
    return np.mean(y_true * y_pred)
def recall_score(y_true, y_pred):
    return np.mean(y_true * y_pred) / np.mean(y_true)
def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r)
class VepleyAiTrain():
    def __init__(self, filesPath = "Datasets",layers = LAYERS,train_ratio = 0.8 ,parentFolder = True,excluded = Excluded) -> None: 
        global LEARNING_RATE, ITERATIONS, ACTIVATION, LAYERS,Actions
        self.start_time = time.time()
        self.date = None
        self.data = None
        self.model_details = {
    "Data": {
        "Train": {
            "Source": [],
            "Size": {"Hand1": 0, "Hand2": 0, "Total":0},
            "Features": ["List of features in training data"],
            "Target": "Target variable or label in training data"
        },
        "Test": {
            "Source": [],
            "Size": {"Hand1": 0, "Hand2": 0, "Total":0},
            "Features": ["List of features in test data"],
            "Target": "Target variable or label in test data"
        },
        "Validation": {
            "Source": [],
            "Size": {"Hand1": 0, "Hand2": 0, "Total":0},
            "Features": ["List of features in validation data"],
            "Target": "Target variable or label in validation data"
        }
    },
    "Time":{
        "Data processing": "Time taken to process data",
        "Training": "Time taken to train model",
        "Total": "Total time taken"
        },
    "Model": {
    "Architecture": "Deep neural network (DNN)",
    "Layers": LAYERS,
    "Activation Functions": ACTIVATION,
    "Training": {
        "Optimizer": "Adam optimizer",
        "Loss Function": "Categorical cross-entropy",
        "Metrics": ["Accuracy"],
        "Batch Size": 32,
        "Epochs": "Number of training epochs"
    },
    "Actions": Actions,
    
    "Result": {
        "Training Loss": "Training loss at the end of training",
        "Training Accuracy": "Training accuracy at the end of training",
        "Validation Loss": "Validation loss at the end of training",
        "Validation Accuracy": "Validation accuracy at the end of training",
        "Test Loss": "Test loss at the end of training",
        "Test Accuracy": "Test accuracy at the end of training",
        "Validation":{
            
            
            
            }
    }
}
}

        #self.read_data(filesPath , parentFolder, excluded)
        
    def read_data(self,path: str,_parentFolder: bool = parent_folder, _excluded: list = Excluded):
        timer = time.time()
        if not os.path.exists(path):
            raise Exception("Path does not exist:" + path)
        #check if detail.json is in path
        self.unpack(path,_parentFolder)
        hand = self.get_details(path,_parentFolder)
        self.data = {'X':np.zeros((len(PARSE_LANDMARKS_JOINTS),NO_OF_SAMPLES)),
                     'Y':np.zeros(NO_OF_SAMPLES),
                     'ID':np.zeros(NO_OF_SAMPLES),'Hand':hand}
        if _parentFolder:
            for folder in os.listdir(path):
                if folder not in _excluded and os.path.isdir(os.path.join(path,folder)):
                    print("Processing dataset: "+folder)
                    self.data = self._append_data(os.path.join(path,folder),self.data)
        else:
            self.data = self._append_data(path,self.data)
        self.data['X'] = self.data['X'].T
        print("Data loaded in "+str(time.time()-timer)+" seconds")
        self.model_details['Time']['Data processing'] = round(time.time()-timer,5)
    def read_validation(self,path: str,_parentFolder: bool = parent_folder):
        if not os.path.exists(path):
            raise Exception("Path does not exist:" + path)
        #check if detail.json is in path
        validation_samples,hand = self.get_validation_details(path)
        self.data['validation'] = {'X':np.zeros((len(PARSE_LANDMARKS_JOINTS),validation_samples)),
                     'Y':np.zeros(validation_samples),
                     'ID':np.zeros(validation_samples),'Hand':hand}
        if _parentFolder:
            for folder in VALIDATION_FILES:
                if  os.path.isdir(os.path.join(path,folder)):
                    print("Processing validation file: "+folder)
                    self.data['validation'] = self._append_data(os.path.join(path,folder),self.data['validation'])
        else:
            self.data['validation'] = self._append_data(path,self.data['validation'])
        self.data['validation']['X'] =self.data['validation']['X'].T
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
    def get_validation_details(self,path):
        validation_samples = 0
        hand = [0,0]
        for i,folder in enumerate(VALIDATION_FILES):
            if os.path.isdir(os.path.join(path,folder)):
                #read detail.json, sum the total
                self.model_details['Data']['Validation']['Source'].append(folder)
                with open(os.path.join(path,folder,"detail.json"),"r") as f:
                    detail = json.load(f)
                    validation_samples +=  detail['hands'][0] + detail['hands'][1]
                    hand[0] += detail['hands'][0]
                    hand[1] += detail['hands'][1]
                    for j in detail.keys():
                        if j == 'width' or j == 'height':
                            continue
        self.model_details['Data']['Validation']['Size']['Total'] = validation_samples
        self.model_details['Data']['Validation']['Size']['Hand1'] = hand[0]
        self.model_details['Data']['Validation']['Size']['Hand2'] = hand[1]
        return validation_samples,hand
    
    def get_details(self,path, parentFolder):
        global NO_OF_SAMPLES,Actions,W,H
        self._transform_data()
        hand = [0,0]
        if parentFolder:
            NO_OF_SAMPLES = 0
            for i,folder in enumerate(os.listdir(path)):
                # Excluded the validation files
                if folder in VALIDATION_FILES:
                    continue
                if os.path.isdir(os.path.join(path,folder)):
                    #read detail.json, sum the total
                    self.model_details['Data']['Train']['Source'].append(folder)
                    self.model_details['Data']['Test']['Source'].append(folder)
                    with open(os.path.join(path,folder,"detail.json"),"r") as f:
                        detail = json.load(f)
                        NO_OF_SAMPLES += detail['hands'][0] + detail['hands'][1]
                        hand[0] += detail['hands'][0]
                        hand[1] += detail['hands'][1]
                        W = detail['width']
                        H = detail['height']
                        for j in detail.keys():
                            if j == 'width' or j == 'height':
                                continue
        else:
            with open(os.path.join(path,"detail.json"),"r") as f:
                detail = json.load(f)
                NO_OF_SAMPLES = detail['hands'][0] + detail['hands'][1]

                W = detail['width']
                H = detail['height']
        return hand
        
        
    def _append_data(self,path: str,data_dict: dict):
        global IGNORE_RIGHT_HAND,FLIP_RIGHT_HAND,PARSE_LANDMARKS_JOINTS
        right_hand_index = data_dict['Hand'][0] 
        if not os.path.exists(path):
            raise Exception("Path does not exist:" + path)
        for j,action in enumerate(Actions):
            print("\rProcessing "+action,f'{j+1} out of {len(Actions)} ',end='')
            with open(os.path.join(path,action+".json"),"r") as f:
                raw_datas = json.load(f)
                for ID in raw_datas:
                    for hand in raw_datas[ID]:
                        if IGNORE_RIGHT_HAND and hand == '1':
                            continue
                        if hand == '0':
                            data_dict['ID'][int(ID)] = int(ID+hand)
                            data_dict['Y'][int(ID)] = j
                        else:
                            data_dict['ID'][int(ID)+right_hand_index] = int(ID+hand)
                            data_dict['Y'][int(ID)+right_hand_index] = j
                        landmarks = {point[0]:np.array(point[1:]) for point in raw_datas[ID][hand]}
                        if FLIP_RIGHT_HAND and hand == '1':
                            for k in landmarks:
                                landmarks[k][0] = W - landmarks[k][0]
                        for i,(_p1,_p2) in enumerate(PARSE_LANDMARKS_JOINTS):
                            if _p1 in landmarks and _p2 in landmarks:
                                tmp = self.calculate_angle(landmarks[_p1],landmarks[_p2])
                                if hand == '0':
                                    data_dict['X'][i][int(ID)] = tmp
                                data_dict['X'][i][int(ID)+right_hand_index] = tmp
                            else:
                                if hand == '0':
                                    data_dict['X'][i][int(ID)] = 0
                                data_dict['X'][i][int(ID)+right_hand_index] = 0
            print("\rProcessed "+action,f'{j+1} out of {len(Actions)} ',end='\n')    
        return data_dict
    def calculate_angle(self,landmark1, landmark2):
        return np.math.atan2(np.linalg.det([landmark1, landmark2]), np.dot(landmark1, landmark2))
    def _transform_data(self):
        '''
        this function is to loop through all the dataFiles, check if details.json have attribute "hands", if not, add it by
        loop through all the {Action}.json files, in each IDs, if have key "0" then count it as hand 0, if have key "1" then count it as hand 1
        '''
        for i,folder in enumerate(os.listdir(dataFiles)):
            if os.path.isdir(os.path.join(dataFiles,folder)):
                with open(os.path.join(dataFiles,folder,"detail.json"),"r") as f:
                    detail = json.load(f)
                    if 'hands' not in detail.keys():
                        detail['hands'] = [0,0]
                        for action in Actions:
                            with open(os.path.join(dataFiles,folder,action+".json"),"r") as f:
                                raw_datas = json.load(f)
                                for ID in raw_datas:
                                    if '0' in raw_datas[ID]:
                                        detail['hands'][0] += 1
                                    if '1' in raw_datas[ID]:
                                        detail['hands'][1] += 1
                        with open(os.path.join(dataFiles,folder,"detail.json"),"w") as f:
                            json.dump(detail,f)
                        print(f'Data {folder} has been updated')
        
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
        pmt_train = pmt[:train_size]
        self.X_test = self.data['X'][train_size:,:]
        self.Y_test = np.eye(len(Actions))[(self.data['Y'][train_size:]).astype(int)]
        #print all the shape
        print("X shape: ",self.X.shape)
        print("Y shape: ",self.Y.shape)
        print("X_test shape: ",self.X_test.shape)
        print("Y_test shape: ",self.Y_test.shape)
        print("X_val shape: ",self.data['validation']['X'].shape)
        print("Y_val shape: ",self.data['validation']['Y'].shape)
        self.model_details['Data']['Train']['Size']['Total'] = self.X.shape[0]
        self.model_details['Data']['Test']['Size']['Total'] = self.X_test.shape[0]
        self.model_details['Data']['Validation']['Size']['Total'] = self.data['validation']['X'].shape[0]
        
        #count the number of pmt_train that < self.data['Hand'][0]
        self.model_details['Data']['Train']['Size']['Hand1'] = np.sum(pmt_train < self.data['Hand'][0])
        self.model_details['Data']['Train']['Size']['Hand2'] = np.sum(pmt_train >= self.data['Hand'][0])
        self.model_details['Data']['Test']['Size']['Hand1'] = self.data['Hand'][0] - self.model_details['Data']['Train']['Size']['Hand1']
        self.model_details['Data']['Test']['Size']['Hand2'] = self.data['Hand'][1] - self.model_details['Data']['Train']['Size']['Hand2']

        self.model_details['Data']['Validation']['Size']['Hand1'] = self.data['validation']['Hand'][0]
        self.model_details['Data']['Validation']['Size']['Hand2'] = self.data['validation']['Hand'][1]
        

        
    def train(self):
        global LAYERS, ITERATIONS, ACTIVATION
        if self.X is None:
            print("No data to train")
            return
        timer = time.time()
        self.model = DNN.create_model(LAYERS,ACTIVATION)
        self.parameters = DNN.train_model(self.model,self.X,self.Y,self.X_test,self.Y_test,ITERATIONS)
        print("Training time: ",time.time()-timer)
        self.model_details['Time']['Training'] = round(time.time()-timer,5)
        
        self.save_parameters()
        #Write to the self.model_details
        self.model_details['Model']['Training']['Epoch'] = ITERATIONS

        
        #test on the validation set
        self.test()
        train_acc = self.parameters.history['accuracy'][-1]
        train_loss = self.parameters.history['loss'][-1]
        test_acc = self.parameters.history['val_accuracy'][-1]
        test_loss = self.parameters.history['val_loss'][-1]
        self.model_details['Model']['Result']['Training Accuracy'] = train_acc
        self.model_details['Model']['Result']['Training Loss'] = train_loss
        self.model_details['Model']['Result']['Test Accuracy'] = test_acc
        self.model_details['Model']['Result']['Test Loss'] = test_loss
        
        DNN.plot_model_history(self.parameters)
        self.predict_random()
        self.model_details['Time']['Total'] = round(time.time()-self.start_time,5)
        print(self.model_details)
        self.save_model_details()

    def save_model_details(self):
        #save as json in script parent folder
        with open('model_details.json', 'w') as outfile:
            s = str(self.model_details).replace("'",'"')
            outfile.write(s)
    def test(self):
        '''
        Using the validation set to run the test
        '''
        X_val = self.data['validation']['X']
        Y_val = self.data['validation']['Y']

        Y_pred = DNN.predict(self.model,X_val)
        Y_pred = np.argmax(Y_pred,axis=1)
        #Write the self.model_details['Model']['Result']["Validation"] consist of all correctly predicted Actions
        for i in range(len(Actions)):
            self.model_details['Model']['Result']['Validation'][Actions[i]] = np.sum(Y_pred[Y_val == i] == i)

        
        
        print("Accuracy: ",accuracy_score(Y_val,Y_pred))
        print("Precision: ",precision_score(Y_val,Y_pred))
        print("Recall: ",recall_score(Y_val,Y_pred))
        print("F1: ",f1_score(Y_val,Y_pred))

        self.model_details['Model']['Result']['Validation Accuracy'] = accuracy_score(Y_val,Y_pred)
        self.model_details['Model']['Result']['Validation Precision'] = precision_score(Y_val,Y_pred)
        self.model_details['Model']['Result']['Validation Recall'] = recall_score(Y_val,Y_pred)
        self.model_details['Model']['Result']['Validation F1'] = f1_score(Y_val,Y_pred)
        

        
        
        
        
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
        pickle.dump({'model':self.model,'parameters':self.parameters},open(os.path.join(processed_folder,dataFiles)+"_parameters.bin","wb"))
        print("Parameters saved to "+processed_folder+dataFiles+"_parameters.bin")
        
        
        
        
if __name__ == "__main__":
    instance = VepleyAiTrain()
    s = "--single" in sys.argv or "-s" in sys.argv
    if len(sys.argv) > 1 :
      if "--process" in sys.argv or "-p" in sys.argv:
        instance.read_data(dataFiles,not s, VALIDATION_FILES)
        instance.read_validation(dataFiles)
        instance.save_processed()
      else:
          instance.load()
      if '--train' in sys.argv or '-t' in sys.argv:
         instance.train()   
    else:
        print("No arguments given, please use --process or --train")
        input("Press enter for debug mode")
        instance.read_data(dataFiles,not s, VALIDATION_FILES)
        instance.read_validation(dataFiles)
        instance.save_processed()
        instance.load()
        instance.train()
      
    
    