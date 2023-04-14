### Di Zhang
### April 5, 2023
### CS5330 - Computer Vision

# cs5330_project5_recognition_using_deep_network

## Project Description

The main object of this project is to train a convolutional neural network that can recognize handwritten digits. In 
addition to that used the transfer training technique to apply network trained on handwritten digits on greek letters by 
changing the last layer of the object. Also, analyzed the weights of the first layer of the network and the images after 
weights being applied. Lastly, made 64 variations of the network and ranked them according to their accuracy in descending
order. For extension, I implemented a live cam handwritten digit recognition program, also added more greek letters to
the dataset.

## How to install and run
### Have the following ready on your **Linux** machine:

- intellij
- python
- cv2
- pytorch
- torchvision
- matplotlib
- other related packages

### How to run

1. open project folder and make sure you folder structure is similar to the one below
2. type in `python3` followed the program you want to run. e.g. `python3 live_cam_digit_recog.py`

### Directory Structure
<pre>
cs5330_project5_recognition_using_deep_network
├── deep_network_recognition
│   ├── data
│   │   └── MNIST
│   │       └── raw
│   │           ├── t10k-images-idx3-ubyte
│   │           ├── t10k-images-idx3-ubyte.gz
│   │           ├── t10k-labels-idx1-ubyte
│   │           ├── t10k-labels-idx1-ubyte.gz
│   │           ├── train-images-idx3-ubyte
│   │           ├── train-images-idx3-ubyte.gz
│   │           ├── train-labels-idx1-ubyte
│   │           └── train-labels-idx1-ubyte.gz
│   ├── experiment.py
│   ├── greek_letter_images_before_covert
│   ├── greek_letter_training.py
│   ├── greek_train
│   │   ├── alpha
│   │   ├── beta
│   │   └── gamma
│   ├── handwritten_digits
│   ├── handwritten_greek_letters
│   │   ├── alpha
│   │   ├── beta
│   │   └── gamma
│   ├── __init__.py
│   ├── live_cam_digit_recog.py
│   ├── main.py
│   ├── network_analysis.py
│   ├── network_test.py
│   ├── __pycache__
│   │   └── main.cpython-310.pyc
│   ├── results
│   │   ├── model.pth
│   │   └── optimizer.pth
│   └── thresholded_digits
├── LICENSE
└── README.md
</pre>

## Program features (matching mechanisms)

#### Basic requirments

- `main.py` - trains a convolutional neural network on 60000 handwritten digits and test on 1000 with an accuracy over 90%
- `network_analysis.py` - print out the 10 weights of the first layer and the images after the weights being applied
- `network_test.py` - test the trained network on self-handwritten digits
- `greek_letter_training.py` - use transfer learning to continue use the already trained network to train on greek letters
- `exeperiment.py` - have a 3-dimensional variation - training data size, drop out rate and activation function - to find the 
optimal training parameters and combination.

#### Extensions
- `live_cam_digit_training.py` - use the network trained before and apply it to a live camera recognitions application
- added more greek letter categories to the training data set and trained the network to recognize more greek letters  

## Demo Video
Extension demo video:
https://youtu.be/S6V5_YvD2Ow

## Time travel day used - 2 day
if you find that I am 2 hours passed the 2 travel days, please forgive me I am on the west coast.