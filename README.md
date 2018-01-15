# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  ![Language](https://img.shields.io/badge/language-Python-green.svg)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./images/CNN-nvidia.png "CNN Structure from NVIDIA Developer Blog"
[image1]: ./images/training-validation.png "Training and Validation Score"
[image2]: ./images/center_2018_01_13_01_05_25_771.jpg "Center Lane Driving"
[image3]: ./images/flip.png "Flipped Image"

[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Rubric Points
 Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* video.mp4 A video recording of the vehicle driving autonomously slightly more than one lap around track 1 (till reaching the bridge for the second time)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the network structure reported by NVIDIA
in their [2016 blog post](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).  A depict of the structure can be seen
below, which was taken from their blog.  In my code, the 
network structure is defined from line 85 to line 121.

![Image0]

It utilizes a Lambda layer to normalize the image, followed by five convolution layers and four dense layers.

Prior to the Lambda layer I used a Cropping2D layer to
crop out the sky (top of the image) and the front part of the car (bottom of the image).
The rest revision I made to it is that I used exponential linear units (ELUs) as activation layers after each 
convolution layer and dense layer. I also used batch normalization to normalize pre-activation outputs. Lastly,
A dropout layer is added in front of all but the last linear dense layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 108, 112, and 116). 

I also trained and validate the model on different sets
of data to check if the model is overfitting (lines 78-82, and 128).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 124).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, smoothly
making turns (lines 62-68 in model.py). 
In center lane driving, I drove counter-clockwisely (forward direction) and clockwisely (reverse direction) for three laps in each direction. 
As for smooth turns, I recorded the how the car make turns
smoothly over the four larger turns (with red and white sides) in both directions.
The data can be downloaded from [here](https://drive.google.com/file/d/1Ltu72L4DduQid8CEf8xw0m95cTyjlx6j/view?usp=sharing). 
I also used the [sample training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
provided on the class website for training and validation (line 71 of the code).


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started using Lenet-5 to drive but it seems difficult 
for the car to make through the first big turn.  I then
switched to [the structure reported by NVIDIA](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) and built models models on it.

My strategy is kind of trial and error.  Oftentimes I
get the result of the car driving out of lane even though the
model seems to be good judging from the training and validation result. Really the decision is kind of fully based
on how the car drives during the autonomous mode.

I tried RELU and ELU for activation layer. It seems that 
ELU provides a faster convergence.  I also looked how the car
would drive when using RELU and it seems that the car would
be moving less smoothly compared to models using ELU.
Batch normalization also seem to improve the convergence speed.

I split the data into training and validation blocks (code line 78) to assess overfitting. The final model I used seems
a bit underfitting, though, as the training score is slightly
higher than the validation score in the figure below. 
![image1]


#### 2. Final Model Architecture

The final model architecture (model.py lines 85-121) consisted of a Cropping2D layer, followed by a 
Lambda layer for normalizing the image, followed by three
convolution blocks with five-by-five kernels and strides of two, followed by two convolution blocks with three-by-three kernels and strides of one, followed by a flatten layer and
four dense blocks.
Each convolution block consists of a convolution layer,
a batch normalization layer and a ELU activation layer.
All dense blocks but the last one has dropout layers
applied to incoming connections. Also all but the last dense blocks uses batch normalization over pre-activation and
utilizes ELU activation. 
Its image was reported above.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps of driving counter-clockwisely (forward) on track one using center lane driving with mouse.
 Here is an example image of center lane driving:

![][image2]

I also recorded three laps of driving clockwisely (reverse)
on track one using center lane driving.
Then I recorded car driving over right or left turning lanes
smoothly over track one. 

I also used sample driving data provided from the course.

To augment the data sat, I also flipped images and angles thinking that this would be negated. For example, here is an image that has then been flipped:

![][image2]
![][image3]

I also used the images from the left and right cameras,
the steering angle would be increased by 0.2 (left camera)
and decreased by 0.2.

This makes the data consists of 92364 images.
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I created a generator for each of the training and validation
data sets. I trained for five epochs with `model.fit_generator()`.  As I mentioned before, it seems that the model is underfitting but it still make through the track.  I will leave the improvement as a future work.
