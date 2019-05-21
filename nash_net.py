import numpy as np
import cv2
import keras
import pickle, os, csv, time, math, configparser, h5py, re, random
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import psutil
import GPUtil
import ast
import hashlib
import base64
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from keras.models import load_model, Sequential, Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import multi_gpu_model
from keras import backend as K

model = Sequential()
model.add(Dense(18, input_dim=18, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(6, activation='linear'))

model.compile(loss='categorical_crossentropy',optimizer='adam')

print(model.summary())