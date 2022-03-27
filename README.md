# Voyage | Visual Speech to text




<!-- #### Demo Video - [Click Here](https://www.youtube.com/watch?v=UnBANdAMgWA) -->

# Introduction

Visual speech recognition,is model created in  Hackathon also known as lip-reading, relies on lip movements to recognise speech without relying on the audio stream. This is particularly useful in noisy environments where the audio signal is corrupted; Automatic lip reading aims to recognise the speech content by watching videos. It has lots of potential applications in both noisy and silent environments and it will help deaf and dumb people to communicate as this system works both ways it has also module which can convert sign language in text with accuracy of 96.5% in visual model.

<table style="display: inline-table;">  
<tr>
<td><img src="https://github.com/chiragarora01/Visual-Speech-To-Text-Team_Voyage/blob/master/screenshot/media1.gif"></td>
<td><img src="https://github.com/chiragarora01/Visual-Speech-To-Text-Team_Voyage/blob/master/screenshot/media3.gif" width="432"></td>
<tr>
<td>0. Input Image(mp4 or Live Camera)</td> 
<td>1. Output</td>
</table>

## What it does

This is a system which predicts what a person is saying on the basis of lip movement of the person,through which one can get an idea what a person is saying and also convert sign language into text.


## Inspiration
In this world which is full of noise, song, speeches, slogans there are few who can't listen to these and this is my small contribution which can make their life easier so that they can easily communicate in this world.

## How we built it
This system is build in 2 parts 
#### 1. Preprocessing 
    
We took all the video and preprocess in 4 steps given below all the videos are from Dataset - [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) which has more than 50,000 video to splitted in train,test and validation which sum up to 500 words of various spekers and after compliting these 4 steps each video it converted into npz file which of numpy arrays to make computation faster:-
<table style="display: inline-table;">  
<tr><td><img src="https://github.com/chiragarora01/Visual-Speech-To-Text-Team_Voyage/blob/master/screenshot/original.gif", width="144"></td><td><img src="https://github.com/chiragarora01/Visual-Speech-To-Text-Team_Voyage/blob/master/screenshot/detected.gif" width="144"></td><td><img src="https://github.com/chiragarora01/Visual-Speech-To-Text-Team_Voyage/blob/master/screenshot/transformed.gif" width="144"></td><td><img src="https://github.com/chiragarora01/Visual-Speech-To-Text-Team_Voyage/blob/master/screenshot/cropped.gif" width="144"></td></tr>
<tr><td>0. Original</td> <td>1. Detection</td> <td>2. Transformation</td> <td>3. Mouth ROIs</td> </tr>
</table>

#### 2. Training  
To train this model we use  articture model where all videos are passed from 3D-CNN after 3DCNN where afterwords it passed into 18 layer of resnet and before passing it to softmax we pushed it to TCN.




## How to get model

Dowload file from this link and place in training folder - [GoogleDrive](https://drive.google.com/file/d/1XTL5ZuipKvKgYQfzYqRcaWAjXsIdHE_Z/view?usp=sharing)


## How to install environment

1. Clone the repository into a directory

```Shell
git clone https://github.com/chiragarora01/Visual-Speech-To-Text-Team_Voyage.git
```

2. Install all required packages.

```Shell
pip install -r requirements.txt
```

## How to Run

To run this script you must have CUDA + CuDnn installed with minimum of 8Gb Ram and a GPU 
```Shell
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Accomplishments that we're proud of
- It take video input,place landmarks on it and predict what it is saying
- It can do this same by live fead.

## What's next for Visual Speech to Text 

To make it a commericaly viable product we need to make some sort of IOT device or to make a system where it take feed from one's device ,pass it to cloud,do computation their and return the desired output as this software need high computation power 
