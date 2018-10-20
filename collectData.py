# Gesture Recognition for controlling the Google Chrome Dino
# Script for Recording the Training data

import cv2
import numpy as np
import os

# Create the Directory Structure
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    os.makedirs('dataset/train_set')
    os.makedirs('dataset/test_set')
    os.makedirs('dataset/train_set/run')
    os.makedirs('dataset/train_set/jump')
    os.makedirs('dataset/test_set/run')
    os.makedirs('dataset/test_set/jump')

    
# Train or Test mode
mode = 'train_set'
directory = 'dataset/' + mode + '/'
    
cap = cv2.VideoCapture(0)

while True:
    
    _, frame = cap.read()
    # Simulating the mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting the count of existing images
    count = {'jump' : len(os.listdir(directory+'jump')),
             'run' : len(os.listdir(directory+'run'))}
    
    # Printing the count for the jump to the screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'JUMP COUNT : '+str(count['jump']), (10,100), font, 1, (0,255,255), 1)
    cv2.putText(frame, 'RUN COUNT : '+str(count['run']), (10,150), font, 1, (0,255,255), 1)

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
    
    cv2.imshow('original', frame)
    
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow('ROI', roi)
    
    interrupt = cv2.waitKey(10)
    
    if interrupt & 0xFF == 27: # ESC Key
        break
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'jump/'+str(count['jump'])+ '.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'run/'+str(count['run'])+ '.jpg', roi)

cv2.destroyAllWindows()
cap.release()
