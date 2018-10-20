# Script to read action, predict and pass actiong to Dino

# Importing the libraries

import numpy as np
from keras.models import model_from_json
import cv2
import os
import pyautogui

# Loading the model
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights('model.h5')
print('Loaded model from disk')

# Initialize capture 
cap = cv2.VideoCapture(0)

while True:
    
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = int(frame.shape[1] - 10)
    y2 = int(0.5*frame.shape[1])

    # Drawing the ROI
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64,64))
    
     # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))

    # Choosing the action for the Dino to perform
    action = 'run'
    if result[0][0] == 0:
        action = 'jump'
    elif result[0][0] == 1:         
        action = 'run'
    
    if action == 'jump':
        pyautogui.keyDown(' ')
        pyautogui.keyUp(' ')
        
    cv2.imshow('test', test_image)
    cv2.putText(frame, 'ACTION : '+action, (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
    cv2.imshow('Frame', frame)
    
    interrupt = cv2.waitKey(10)
    
    if interrupt & 0xFF == 27: # ESC Key
        break
    
cv2.destroyAllWindows()
cap.release()