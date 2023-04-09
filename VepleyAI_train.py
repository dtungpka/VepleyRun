
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
import datetime
from VepleyAI_acquire import Actions
from tkinter import filedialog
import openpyxl
TRAIN_VALIDATION_RATIO = 0.8
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
#the first value is the number of features
LAYERS = [
#[32, 8,len(Actions)],
#[64, 32, len(Actions)],
#[128, 64, 32, len(Actions)],
#[256, 128, 64, 32, len(Actions)],
[512, 256, 128, 64, 32, len(Actions)],
[1024, 512, 256, 128, 64, 32, len(Actions)],
[2048, 1024, 512, 256, 128, 64, 32, len(Actions)],
[4096, 2048, 1024, 512, 256, 128, 64, 32, len(Actions)]
]
    
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
TEST_SET = ['Hung']
VALIDATION_SET = ['Tung']
test_folders = []
validation_folders = []

cf_matrix = {'TP':None, 'FP': None, 'TN': None, 'FN': None}
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


def write_results_to_excel(table_name: str, results_dict: dict, spacing: bool = False):
    """
    Write a table with the given table_name and results_dict to an excel file named 'results.xlsx'.
    If the file already exists, append the table to a new row.
    
    Args:
    - table_name (str): The name of the table to write to the excel file.
    - results_dict (dict): A dictionary containing the results to write to the excel file.
      This should contain the keys 'TP', 'FP', 'TN', and 'FN'.
    - spacing (bool): If True, add a blank line before adding a new row.
    """
    try:
        # Load the workbook if it exists
        workbook = openpyxl.load_workbook(os.path.join(processed_folder,'results.xlsx'))
    except FileNotFoundError:
        # Create a new workbook if it doesn't exist
        workbook = openpyxl.Workbook()
        # Create the header for the columns
        header = ['Label', 'True Positive', 'False Positive', 'True Negative', 'False Negative']
        # Select the active worksheet and write the header
        worksheet = workbook.active
        worksheet.append(header)

    # Select the active worksheet
    worksheet = workbook.active
    
    # Add a blank line if spacing is True
    if spacing:
        worksheet.append([])

    # Write the table name and results to the worksheet
    row = [table_name, results_dict['TP'], results_dict['FP'], results_dict['TN'], results_dict['FN']]
    worksheet.append(row)

    # Save the changes to the workbook
    workbook.save(os.path.join(processed_folder,'results.xlsx'))




def get_test_val_set_name(data_files_path: str) -> tuple:
    """
    This function takes a path to the directory containing the dataset folders and returns two lists of folder names.
    It loops through all the folders in the directory and checks if they have a name that contains any of the strings in
    TEST_SET or VALIDATION_SET. If a folder matches a name in TEST_SET, its name is added to test_folders. If it matches
    a name in VALIDATION_SET, its name is added to validation_folders.
    
    Args:
    - data_files_path (str): Path to the directory containing the dataset folders.
    
    Returns:
    A tuple of two lists: test_folders and validation_folders.
    """
    test_folders = []
    validation_folders = []
    
    for folder_name in os.listdir(data_files_path):
        if folder_name.startswith('VepleyAI_dataset_'):
            if any(name in folder_name for name in TEST_SET):
                test_folders.append(folder_name)
            elif any(name in folder_name for name in VALIDATION_SET):
                validation_folders.append(folder_name)
                
    return test_folders, validation_folders

    
    
class VepleyAiTrain():
    def __init__(self, filesPath = "Datasets",layers = LAYERS,train_ratio = 0.8 ,parentFolder = True,excluded = Excluded) -> None: 
        global LEARNING_RATE, ITERATIONS, ACTIVATION, LAYERS,Actions
        self.start_time = time.time()
        self.date = None
        self.data = {}
        self.data['Train'] = None
        self.data['Validation'] = None
        self.data['Test'] = None
        self.models = {}
        self.parameters = {}
        self.model_details = {
    "Data": {
        "Train": {
            "Source": [],
            "Size": {"Hand1": 0, "Hand2": 0, "Total":0},
            "Features": ["List of features in training data"],
            "Target": "Target variable or label in training data"
        },
        "Validation": {
            "Source": [],
            "Size": {"Hand1": 0, "Hand2": 0, "Total":0},
            "Features": ["List of features in validation data"],
            "Target": "Target variable or label in validation data"
        },
        "Test": {
            "Source": [],
            "Size": {"Hand1": 0, "Hand2": 0, "Total":0},
            "Features": ["List of features in test data"],
            "Target": "Target variable or label in test data"
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
    
    "Result": []
}
}

        #self.read_data(filesPath , parentFolder, excluded)
        
    def read_data(self,path: str,_parentFolder: bool = parent_folder, _excluded: list = Excluded):
        timer = time.time()
        if not os.path.exists(path):
            raise Exception("Path does not exist:" + path)
        #check if detail.json is in path
        self.unpack(path,_parentFolder)
        hand = self._get_details(path,_parentFolder)
        self.data['Train'] = {'X':np.zeros((len(PARSE_LANDMARKS_JOINTS),NO_OF_SAMPLES)),
                     'Y':np.zeros(NO_OF_SAMPLES),
                     'ID':np.zeros(NO_OF_SAMPLES),'Hand':hand}
        if _parentFolder:
            for folder in os.listdir(path):
                if folder not in _excluded and os.path.isdir(os.path.join(path,folder)):
                    print("Processing dataset: "+folder)
                    self.data['Train'] = self._append_data(os.path.join(path,folder),self.data['Train'])
        else:
            self.data['Train'] = self._append_data(path,self.data['Train'])
        self.data['Train']['X'] = self.data['Train']['X'].T
        print("Data loaded in "+str(time.time()-timer)+" seconds")
        self.model_details['Time']['Data processing'] = round(time.time()-timer,5)
    
    def read_validation(self,path: str,_parentFolder: bool = parent_folder):
        global validation_folders
        if not os.path.exists(path):
            raise Exception("Path does not exist:" + path)
        #check if detail.json is in path
        validation_samples,hand,self.model_details['Data']['Validation'] = self.get_details(path,validation_folders,self.model_details['Data']['Validation'])#_samples,hand,details
        self.data['Validation'] = {'X':np.zeros((len(PARSE_LANDMARKS_JOINTS),validation_samples)),
                     'Y':np.zeros(validation_samples),
                     'ID':np.zeros(validation_samples),'Hand':hand}
        if _parentFolder:
            for folder in validation_folders:
                if  os.path.isdir(os.path.join(path,folder)):
                    print("Processing validation file: "+folder)
                    self.data['Validation'] = self._append_data(os.path.join(path,folder),self.data['Validation'])
        else:
            self.data['Validation'] = self._append_data(path,self.data['Validation'])
        self.data['Validation']['X'] =self.data['Validation']['X'].T
    def read_test(self,path: str,_parentFolder: bool = parent_folder):
        global test_folders
        if not os.path.exists(path):
            raise Exception("Path does not exist:" + path)
        #check if detail.json is in path
        test_samples,hand,self.model_details['Data']['Test'] = self.get_details(path,test_folders,self.model_details['Data']['Test'])#_samples,hand,details
        self.data['Test'] = {'X':np.zeros((len(PARSE_LANDMARKS_JOINTS),test_samples)),
                     'Y':np.zeros(test_samples),
                     'ID':np.zeros(test_samples),'Hand':hand}
        if _parentFolder:
            for folder in test_folders:
                if  os.path.isdir(os.path.join(path,folder)):
                    print("Processing test file: "+folder)
                    self.data['Test'] = self._append_data(os.path.join(path,folder),self.data['Test'])
        else:
            self.data['Test'] = self._append_data(path,self.data['Test'])
        self.data['Test']['X'] =self.data['Test']['X'].T
    def save_processed(self):
        global processed_folder
        if not os.path.exists(processed_folder):
            os.mkdir(processed_folder)
        dt = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d_%H-%M-%S")
        pickle.dump(self.data,open(os.path.join(processed_folder,f"processed_{dt}")+".VDP","wb"))
        print("Data processed,saved to "+processed_folder+".VDP")
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
    def get_details(self,path,set_list,details):
        _samples = 0
        hand = [0,0]
        for i,folder in enumerate(set_list):
            if os.path.isdir(os.path.join(path,folder)):
                #read detail.json, sum the total
                details['Source'].append(folder)
                with open(os.path.join(path,folder,"detail.json"),"r") as f:
                    detail = json.load(f)
                    _samples +=  detail['hands'][0] + detail['hands'][1]
                    hand[0] += detail['hands'][0]
                    hand[1] += detail['hands'][1]
        details['Size']['Total'] = _samples
        details['Size']['Hand1'] = hand[0]
        details['Size']['Hand2'] = hand[1] #['Data']['Test']
        return _samples,hand,details
    
    def _get_details(self,path, parentFolder):
        global NO_OF_SAMPLES,Actions,W,H
        self._transform_data()
        hand = [0,0]
        if parentFolder:
            NO_OF_SAMPLES = 0
            for i,folder in enumerate(os.listdir(path)):
                # Excluded the test files
                if folder in test_folders:
                    continue
                if folder in validation_folders:
                    continue
                if os.path.isdir(os.path.join(path,folder)):
                    #read detail.json, sum the total
                    self.model_details['Data']['Train']['Source'].append(folder)
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
        right_hand_index = data_dict['Hand'][0] -1
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
                            right_hand_index += 1
                            data_dict['ID'][right_hand_index] = int(ID+hand)
                            data_dict['Y'][right_hand_index] = j
                        landmarks = {point[0]:np.array(point[1:]) for point in raw_datas[ID][hand]}
                        if FLIP_RIGHT_HAND and hand == '1':
                            for k in landmarks:
                                landmarks[k][0] = W - landmarks[k][0]
                        for i,(_p1,_p2) in enumerate(PARSE_LANDMARKS_JOINTS):
                            if _p1 in landmarks and _p2 in landmarks:
                                tmp = self.calculate_angle(landmarks[_p1],landmarks[_p2])
                                if hand == '0':
                                    data_dict['X'][i][int(ID)] = tmp
                                else:
                                    data_dict['X'][i][right_hand_index] = tmp
                            else:
                                if hand == '0':
                                    data_dict['X'][i][int(ID)] = 0
                                else:
                                    data_dict['X'][i][right_hand_index] = 0
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
        if self.data['Train'] == None:
            #prompt to select file, end with .bin
            global processed_folder
            filename = filedialog.askopenfilename(initialdir = processed_folder,title = "Select file",filetypes = (("Vepley data processed files","*.VDP"),("all files","*.*")))
            if filename:
                self.data = pickle.load(open(filename,"rb"))
                print("Data loaded from "+filename)
            
        
        #spilt the data
        self._split_data()
    def _split_data(self):
        global NO_OF_SAMPLES
        pmt = np.random.permutation(NO_OF_SAMPLES)
        #self.data['Train']['X'] = self.data['Train']['X'][pmt,:]
        #self.data['Train']['Y'] = self.data['Train']['Y'][pmt]
        
        self.X = self.data['Train']['X']
        self.Y = self.data['Train']['Y']
        #from Y (800,) to (800,5)
        self.Y = np.eye(len(Actions))[self.Y.astype(int)]
        self.X_val = self.data['Validation']['X']
        self.Y_val = np.eye(len(Actions))[(self.data['Validation']['Y']).astype(int)]

        self.data['Test']['Y'] = np.eye(len(Actions))[(self.data['Test']['Y']).astype(int)]
        #print all the shape
        print("X shape: ",self.X.shape)
        print("Y shape: ",self.Y.shape)
        print("X_val shape: ",self.X_val.shape)
        print("Y_val shape: ",self.Y_val.shape)
        print("X_test shape: ",self.data['Test']['X'].shape)
        print("Y_test shape: ",self.data['Test']['Y'].shape)

        self.model_details['Data']['Train']['Size']['Total'] = self.X.shape[0]
        self.model_details['Data']['Validation']['Size']['Total'] = self.X_val.shape[0]
        self.model_details['Data']['Test']['Size']['Total'] = self.data['Test']['X'].shape[0]
        
        #count the number of pmt_train that < self.data['Hand'][0]
        for s in ('Train','Validation', 'Test'):
            self.model_details['Data'][s]['Size']['Hand1'] = self.data[s]['Hand'][0]
            self.model_details['Data'][s]['Size']['Hand2'] = self.data[s]['Hand'][1]
        
        

        
    def train(self,layers):
        global LAYERS, ITERATIONS, ACTIVATION
        if self.X is None:
            print("No data to train")
            return
        timer = time.time()
        print(f"\nTraining {layers}")
        model = DNN.create_model(layers,ACTIVATION)
        parameters = DNN.train_model(model,self.X,self.Y,self.X_val,self.X_val,ITERATIONS) #history
        #print the model.summary()
        model.summary()
        print("Training time: ",time.time()-timer)
        self.model_details['Time']['Training'] = round(time.time()-timer,5)
        #parse datetime in hhmmssttddmmyy
        dt = datetime.now().strftime('%H%M%S%d%m%y')
        name = f"{parameters.history['val_accuracy'][-1]}_{dt}.VMD"
        self.save_parameters(model,parameters,name)
        #Write to the self.model_details
        self.model_details['Model']['Training']['Epochs'] = ITERATIONS
        train_acc = self.parameters.history['accuracy'][-1]
        train_loss = self.parameters.history['loss'][-1]
        val_acc = self.parameters.history['val_accuracy'][-1]
        val_loss = self.parameters.history['val_loss'][-1]
        #create a cf = {'TP':None, 'FP': None, 'TN': None, 'FN': None} for the valiation set, for each label
        for i in range(len(Actions)):
            cf = {'TP':None, 'FP': None, 'TN': None, 'FN': None}
            cf['TP'] = np.sum(np.logical_and(self.Y_val[:,i] == 1, self.predictions[:,i] == 1))
            cf['FP'] = np.sum(np.logical_and(self.Y_val[:,i] == 0, self.predictions[:,i] == 1))
            cf['TN'] = np.sum(np.logical_and(self.Y_val[:,i] == 0, self.predictions[:,i] == 0))
            cf['FN'] = np.sum(np.logical_and(self.Y_val[:,i] == 1, self.predictions[:,i] == 0))
            ex_name = f'Action {i} Validation using {layers}'
            write_results_to_excel(ex_name,cf,i == 0)
       
        m_details = {}
        m_details['Layers'] = layers
        m_details['Training Accuracy'] = train_acc
        m_details['Training Loss'] = train_loss
        m_details['Validation Accuracy'] = val_acc
        m_details['Validation Loss'] = val_loss
        m_details['Time']= round(time.time()-timer,5)
        DNN.plot_model_history(parameters)
        self.predict_random(model)
        self.model_details['Time']['Total'] = round(time.time()-self.start_time,5)
        print(self.model_details)
        self.model_details['Result'].append(m_details)
        self.save_model_details()
        self.test(model,layers)

    def save_model_details(self):
        #save as json in script parent folder
        with open('model_details.json', 'w') as outfile:
            s = str(self.model_details).replace("'",'"')
            outfile.write(s)
    def test(self,model,layers):
        '''
        Using the test set to run the test
        '''
        X_test = self.data['Test']['X']
        Y_test = self.data['Test']['Y']

        Y_pred = DNN.predict(model,X_test)
        Y_pred = np.argmax(Y_pred,axis=1)
        #create a cf = {'TP':None, 'FP': None, 'TN': None, 'FN': None} for the test set, for each label
        for i in range(len(Actions)):
            cf = {'TP':None, 'FP': None, 'TN': None, 'FN': None}
            cf['TP'] = np.sum(np.logical_and(Y_test[:,i] == 1, Y_pred[:,i] == 1))
            cf['FP'] = np.sum(np.logical_and(Y_test[:,i] == 0, Y_pred[:,i] == 1))
            cf['TN'] = np.sum(np.logical_and(Y_test[:,i] == 0, Y_pred[:,i] == 0))
            cf['FN'] = np.sum(np.logical_and(Y_test[:,i] == 1, Y_pred[:,i] == 0))
            ex_name = f'Action {i} Test using {layers}'
            write_results_to_excel(ex_name,cf,False)
        

        
        
        
        
    def predict_random(self,model):
        if self.X_test is None:
            print("No data to predict")
            return
        for i in range(0,10):
            index = random.randint(0,self.X_test.shape[1])
            X_pred = self.X_test[index,:]
            Y_pred = DNN.predict(model,X_pred.reshape(1,20))
            print("Predicted: ",Actions[np.argmax(Y_pred)])
            Y_true = self.Y_test[index,:]
            print("True: ",Actions[np.argmax(Y_true)])


    def save_parameters(self,model, parameter, name):
        """
        Save the model and parameter objects as a dictionary to a file in the processed folder.
    
        Args:
            model (object): Class instance representing the trained model.
            parameter (object): Class instance representing the model's hyperparameters.
            name (str): Name to be given to the file. It should end with .pkl.
        """
        # Create a dictionary containing the model and parameter objects
        data = {"model": model, "parameter": parameter}
    
        # Create the full path to the output file
        output_file = os.path.join(processed_folder, name)
    
        # Save the dictionary to the output file
        with open(output_file, "wb") as f:
            pickle.dump(data, f)
        
        print("Saved model and parameters to file:", output_file)

        
        
def train_layers(layers,instance):
    print("There are", len(layers), "layers in the model.")
    choice = input("Do you want to train all layers? (y/n) ")
    if choice.lower() == 'y':
        for layer in layers:
            instance.train(layer)
    else:
        while True:
            index = input("Which layer do you want to train? Enter the index (starting from 0) or 'q' to quit. ")
            if index.lower() == 'q':
                break
            try:
                index = int(index)
                if index < 0 or index >= len(layers):
                    print("Invalid index. Please enter a number between 0 and", len(layers)-1)
                    continue
                instance.train(layers[index])
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")

    
        
        
if __name__ == "__main__":
    test_folders, validation_folders = get_test_val_set_name(dataFiles)
    instance = VepleyAiTrain()
    s = "--single" in sys.argv or "-s" in sys.argv
    if len(sys.argv) > 1 :
      if "--process" in sys.argv or "-p" in sys.argv:
        instance.read_data(dataFiles,not s, test_folders + validation_folders)
        instance.read_validation(dataFiles)
        instance.read_test(dataFiles)
        instance.save_processed()
      else:
          instance.load()
      if '--train' in sys.argv or '-t' in sys.argv:
         train_layers(LAYERS, instance)  
    else:
        print("No arguments given, please use --process or --train")
        i = input("Press enter for debug mode or t to train: ")
        if 't' in i:
            instance.load()
            train_layers(LAYERS, instance)
        else:
            instance.read_data(dataFiles,not s, test_folders + validation_folders)
            instance.read_validation(dataFiles)
            instance.read_test(dataFiles)
            instance.save_processed()
            instance.load()
            train_layers(LAYERS, instance)  
      
    
    