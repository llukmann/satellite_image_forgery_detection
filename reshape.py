# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:43:39 2022

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
from ELA import *
from Preparing_dataset import*
from class_labels import *

x_casia = x_casia.reshape(-1, 128, 128, 3)
y_casia = to_categorical(y_casia, 2) #y is one hot encoded