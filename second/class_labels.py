# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:07:02 2022

@author: lukma
"""

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
from Preparing_dataset import*

dataset.to_csv('final_valid_BST.csv',index=False)
dataset = pd.read_csv('final_valid_BST.csv')

x_casia = []
y_casia = []

for index, row in tqdm(dataset.iterrows()):
    x_casia.append(np.array(ELA(row[0]).resize((128, 128))).flatten() / 255.0)
    y_casia.append(row[1])
    
x_casia = np.array(x_casia)
y_casia = np.array(y_casia)

print (len(x_casia))

x_casia = x_casia.reshape(-1, 128, 128, 3)
y_casia = to_categorical(y_casia, 2) #y is one hot encoded

from numpy import save
## save all the data 
save('X_final_valid_BST.npy', x_casia)
save('Y_final_valid_BST.npy',y_casia)



