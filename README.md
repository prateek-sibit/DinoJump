# DinoJump
OpenCV and Convolutional Neural Networks with Gesture Detection for playing the Dino Game in Chrome

## About the Files in this Project

- dataset : Contains two subfolders train_set and test_set, each further containing sub folders run and jump containing the images for 
training and testing of the CNN Model

- collectData.py : Script for collecting the training and testing data on which the model is to be later trained

- model.h5 : CNN model created by Me stored in .h5 format

- model.json : CNN model created by Me stored in .json format

- model.py : Keras Convolutional Neural Network Model that is used for prediction

- predict.py : Main script to be run after the data has been collected and model has been trained. Used to play the Game

## How to Use The Project :

1. Clone the Repository into your directory of choice
2. Run collectData.py 
  - The Region of Interest (ROI) is the Area in the top right corner of the screen which as a green boundary
  - Two hand gestures are needed for controlling the Dino that are "Run" and "Jump"
  - "Run" is Recorded by pressing "1" on your Keyboard, "Jump" is recorded by pressing "2" on your keyboard
  - Each keypress will show you the updated number of recorded images for both Run and Jump on the screen
  - I used 200 images for both Run and Jump, the number you choose is on you
3. After collecting these images run model.py, this trains a CNN Model on these collected images
4. After your model has been trained you can run predict.py to actually play the game
  - The Dino will respond to (Run/Jump) on the gestures you have trained it on
  
  



