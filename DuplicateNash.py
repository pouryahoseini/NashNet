import numpy as np
from numpy import array
import cv2
import nashpy
import ast
import pandas as pd

def GetTrainingDatafromCSV(filename):
    '''
    Function to read input CSV files
    '''

    df = pd.read_csv(filename, sep=',', converters={'Game': ast.literal_eval, 'Nash': ast.literal_eval})
    games = df.Game.tolist()
    equilibria = df.Nash.tolist()

    return games, equilibria


def Duplicate(array):
    newarray = []
    count = 0
    for i in range(0,10):
        if i < len(array):
            newarray.append(array[i])
        else:
            if count < len(array):
                newarray.append(array[count])
                count = count + 1
            else:
                count = 0
                newarray.append(array[count])
                count = count + 1
    return newarray

def WriteToCSV(array):
    DUPLICATENASH_FILENAME = '/home/tapadhird/Desktop/newnash.csv'
    df = pd.DataFrame(columns=["Nash"])
    dict_list = []
    for i in range(0, array.size):
        d = {"Nash":array[i]}
        dict_list.append(d)
    df2 = df.append(dict_list)
    df2.to_csv(DUPLICATENASH_FILENAME)
    return

def ConvertToTen(array):
    size = 0
    for i in range(array.size):
        #print(array[i])
        size = len(array[i])
        if size < 10:
            array[i] = Duplicate(array[i])
    WriteToCSV(array)
    return

TRAINING_FILENAME = '/home/tapadhird/Desktop/Training.csv'


trainingSamples, trainingEqs2 = GetTrainingDatafromCSV(TRAINING_FILENAME)

trainingEqs2 = np.array(trainingEqs2)
ConvertToTen(trainingEqs2)






