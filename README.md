# Emergency-Signal-Recognition-for-the-Hearing-Impaired-using-Multichannel-CNN

We live in a world where auditory means are used to warn people of Emergency situations. For the hearing impaired, comprehending such auditory signals becomes impossible. So this system aims to notify the hearing imapired of the emergency situations by identifying emergency signals such as Police sirens, Ambulance siren, Fire alarms. The task is classify the background audio signal as emergency or non-emergency. This model can be deployed in an Android app that vibrates to notify the hearing impaired in the case of an emergency. Further this can be deployed on a smart-watch or a fitness band.

## Dataset

The dataset used was Google's Audioset. A part of Audioset was used for this study, out of the 632 event classes present in Audioset only 35 classes were used. Classes such as Sirens, alarms, buzzers were marked as emergency while sounds of wind, traffic noises, rain, etc. were marked as non-emergency. The file data_aquisition.py consists of the python script for extracting the 10 second audio-clips from the video links in Audioset csv file. To extract the data run that script, and change the target folder to the desired directory. 

## Data Preprocessing 

- Converting raw audio clips to Mel spectrograms by making use of Librosa library.
- Using traditional audio augmentation techniques such as adding white noise, time stretching, time shifting to increase the size of training data.
- Using Mixup augmentation technique for effectively generating new clips from existing ones.

## Learning Model

A multichannel CNN architecture was implemented, which made use of 4 different channels to extract features from spectrograms, and then the 4 channels are merged using the Add() layer in Keras. 

## Requirements

- Python 3
- Numpy 
- Librosa
- Keras
- Matplotlib
- Seaborn
- Sklearn
