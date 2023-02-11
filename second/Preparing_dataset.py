# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:04:51 2022

@author: lukma
"""
#Importing all the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm
import random
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image, ImageChops, ImageEnhance
from IPython.display import display # to display images
from ELA import ELA

path_original = 'C:/Users/lukma/Desktop/DS/BST/valid/real/'
path_tampered = 'C:/Users/lukma/Desktop/DS/BST/valid/fake/'

total_original = os.listdir(path_original)
total_tampered = os.listdir(path_tampered)

images = []
r = 0
for file in tqdm(os.listdir(path_original)):
    try:
        line = path_original + file  + ',0\n'
        images.append(line)
        r+=1
        if (r > 504): break
    except:
        print(path_original+file)
        
k = 0
for file in tqdm(os.listdir(path_tampered)):
    try:
        line = path_tampered + file + ',1\n'
        images.append(line)
        k+=1
        if (k > 504): break
    except:
          print(path_tampered+file)
          
image_name = []
label = []
for i in tqdm(range(len(images))):
    image_name.append(images[i][0:-3])
    label.append(images[i][-2])
    
dataset = pd.DataFrame({'image':image_name,'class_label':label})

print (len (images))