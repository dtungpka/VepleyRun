
#import mediapipe as mp
#import cv2
#import time
#import numpy as np
#import win32api, win32con


#cap = cv2.VideoCapture(0)

#mpHands = mp.solutions.hands
##confiden = 0.4
#hands = mpHands.Hands(min_detection_confidence=0.3,
#               min_tracking_confidence=0.3)
#mpDraw = mp.solutions.drawing_utils
#lastMousePos = (0,0)

#def MoveMouse(pos):
#    #smooth the mouse movement from lastMousePos to pos
#    global lastMousePos
#    x,y = pos
#    lastX,lastY = lastMousePos
#    for i in range(1,10):
#        win32api.SetCursorPos((int(lastX+i*(x-lastX)/10),int(lastY+i*(y-lastY)/10)))
#        time.sleep(0.01)
#        lastMousePos = pos
        

#while True:
#    success, img = cap.read()

#    #convert the image to RGB
#    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    #process the image
#    results = hands.process(imgRGB)
#    #print(results.multi_hand_landmarks)
#    if results.multi_hand_landmarks:
#        for handLms in results.multi_hand_landmarks:
#            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#            #draw a cirle on landmark 8
#            for id, lm in enumerate(handLms.landmark):
#                h, w, c = img.shape
#                cx, cy = int(lm.x * w), int(lm.y * h)
#                print(id, cx, cy)
#                if id == 8:
#                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
#                    MoveMouse(((w-cx)*2,cy*2))
#    cv2.imshow("Image", img)
#    cv2.waitKey(1)
import numpy as cp
#import cupy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class NeuralNetwork:
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes

    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))
    
    def nn(self, weights, inputs):
        layer1 = self.sigmoid(cp.dot(inputs, self.weights1))
        output = self.sigmoid(cp.dot(layer1, self.weights2))
        return output

    def loss(self, weights, inputs, targets):
        predictions = self.nn(weights, inputs)
        error = cp.mean((predictions - targets)**2)
        return error

    def fit(self, inputs, targets):
        self.weights1 = cp.random.rand(self.num_features, 20)
        self.weights2 = cp.random.rand(20, self.num_classes)

        result = minimize(self.loss, [self.weights1, self.weights2], args=(inputs, targets), method='BFGS')
        self.trained_weights = result.x
        self.loss_history = result.fun
    
    def predict(self, inputs):
        layer1 = self.sigmoid(cp.dot(inputs, self.weights1))
        output = self.sigmoid(cp.dot(layer1, self.weights2))
        return cp.argmax(output, axis=1)
    
    def show_loss_graph(self):
        plt.plot(self.loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss over iterations")
        plt.show()

# Generate random data for training
inputs = cp.random.rand(1000, 20)
targets = cp.random.randint(0, 5, size=(1000,))

# Initialize the neural network model
model = NeuralNetwork(num_features=20, num_classes=5)

# Train the model
model.fit(inputs, targets)

# Plot the loss over the iterations
model.show_loss_graph()