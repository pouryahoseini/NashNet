from __future__ import absolute_import, division, print_function, unicode_literals

#Headers
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import contrib
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import ast
import pandas as pd

import pickle, os, csv, time, math, h5py, re, random
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import psutil
import hashlib
import base64


autograph = contrib.autograph

'''****************************'''
'''Global variables'''
PLAYER_NUMBER = 2
PURE_STRATEGIES_PER_PLAYER = 3
WEIGHT_EQUILIBRIA_DISTANCE = 1
WEIGHT_PAYOFF_DIFFERENCE = 1
TRAINING_FILENAME = 'Training.csv'


'''****************************'''
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
    return np.array(newarray)

def WriteToCSV(array):
    DUPLICATENASH_FILENAME = './gennash.csv'
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
    return array

def GetTrainingDataFromNPY(data_file, labels_file):
    data = np.load(data_file)
    labels = np.load(labels_file)

    return data, labels

def main():
    '''
    Main function
    '''
    # trainingEq_numpy = np.zeros((993,10,2,3))
    
    #Read the training data
    #Indexing:
    #trainingSamples: [sample #] [player #] [row # of the game] [column # of the game]
    #trainingEqs: [sample #] [equilibrium #] [player #] [pure strategy #]
    # trainingSamples, trainingEqs2 = GetTrainingDatafromCSV(TRAINING_FILENAME)

    # trainingEqs=[]
    # for i in range(len(trainingEqs2)):
    #     trainingEqs.append(trainingEqs2[i][0])
        
    # trainingEqs = np.array(trainingEqs)
    # trainingSamples = np.array(trainingSamples)

    # trainingEqs2 = np.array(trainingEqs2)
    # conv_nash = ConvertToTen(trainingEqs2)
    # print("\n\n\n\n\n\n\n\n\n\n\n\n")
    # print(conv_nash.shape)
    # print(conv_nash[0].shape)
    # print("\n\n\n\n\n\n\n\n\n\n\n\n")
#     trainingEqs = tf.data.Dataset.from_tensor_slices(trainingEqs)
#     trainingSamples = tf.data.Dataset.from_tensor_slices(trainingSamples)
    #print(trainingEqs.output_shapes)
    trainingSamples, trainingEqs = GetTrainingDataFromNPY("Training.npy","Training-Labels.npy")
    
    #Constructing the neural network
    #Input layer
    nn_Input = layers.Input(shape = (PLAYER_NUMBER, PURE_STRATEGIES_PER_PLAYER, PURE_STRATEGIES_PER_PLAYER))
    flattenedInput = layers.Flatten(input_shape=(PLAYER_NUMBER, PURE_STRATEGIES_PER_PLAYER, PURE_STRATEGIES_PER_PLAYER))(nn_Input)
    
    #Fully-connected layers
    layer1 = layers.Dense(200, activation = 'relu')(flattenedInput)
    layer2 = layers.Dense(500, activation = 'tanh')(layer1)
    layer3 = layers.Dense(500, activation = 'relu')(layer2)
    layer4 = layers.Dense(500, activation = 'relu')(layer3)
    layer5 = layers.Dense(500, activation = 'tanh')(layer4)
    layer6 = layers.Dense(500, activation = 'relu')(layer5)
    layer7 = layers.Dense(500, activation = 'relu')(layer6)
    layer8 = layers.Dense(200, activation = 'tanh')(layer7)
    layer9 = layers.Dense(100, activation = 'relu')(layer8)
    lastLayer_player1 = layers.Dense(PURE_STRATEGIES_PER_PLAYER)(layer9)
    lastLayer_player2 = layers.Dense(PURE_STRATEGIES_PER_PLAYER)(layer9)
    
    #Softmax layers. Two for the two players
    softmax1 = layers.Activation('softmax')(lastLayer_player1)
    softmax2 = layers.Activation('softmax')(lastLayer_player2)
    
    #Output layer
    concatenated_output = layers.concatenate([softmax1, softmax2])
    nn_Single_Out = layers.Reshape((PLAYER_NUMBER, PURE_STRATEGIES_PER_PLAYER))(concatenated_output)
    nn_cSingle = layers.concatenate([nn_Single_Out, nn_Single_Out, nn_Single_Out, nn_Single_Out, nn_Single_Out, 
                                    nn_Single_Out, nn_Single_Out, nn_Single_Out, nn_Single_Out, nn_Single_Out])
    nn_Output = layers.Reshape((10,2,3))(nn_cSingle)
    
    #Defining the NN model
    nn_model = keras.Model(inputs = nn_Input, outputs = nn_Output)
    
    # Create the optimizer
    keras.optimizers.SGD(learning_rate=0.1)

    #Compiling the NN model
    nn_model.compile(   loss = lossFunction(nn_Input, WEIGHT_EQUILIBRIA_DISTANCE, WEIGHT_PAYOFF_DIFFERENCE), 
                        optimizer = 'adam', 
                        metrics = ['mse'])

    #Printing the summary of the constructed model
    print(nn_model.summary())
    
    #Train the NN model
    nn_model.fit(trainingSamples, trainingEqs, epochs = 100, batch_size = 32)


def get_payoff(game, nash, num_options):
    #Get equations for each player, and reshape them so that when they are multiplied together
    # it creats a num_options by num_options matrix of the probabilities that a given payoff
    # will be chosen
    e_0 = K.gather(K.flatten(nash), K.arange(start=0,stop=3, step=1))
    e_0 = K.reshape(e_0, (num_options,1))
    e_1 = K.gather(K.flatten(nash), K.arange(start=3,stop=6, step=1))
    e_1 = K.reshape(e_0, (1,num_options))
    # e_1 = K.gather(nash, K.constant([1], dtype='int32'))
    # e_1 = K.reshape(e_1, (1,num_options))

    #Multiply them together to get the matrix
    probability_mat = e_0 * e_1

    #Reshape the matrix, so that concatenating with axis 0 produces the right thing
    probability_mat = K.reshape(probability_mat, (1, num_options, num_options))

    #Concatenate probability mat with itself to get a tensor with shape (2, num_options, num_options)
    #   This is explicit for 2 players. Adding more will be more complicated and I don't want to 
    #   deal with that right now
    probability_mat = K.concatenate([probability_mat, probability_mat], axis=0)

    #Multiply the probability matrix by the game (payoffs) to get the expected payoffs
    #   for each player
    exp_payoff_mat = game * probability_mat

    #Sum the expected payoff matrix for each player (eg: (2,3,3)->(2,1))
    payoffs = K.sum(K.sum(exp_payoff_mat, axis=1), axis=1)

    return payoffs


'''****************************'''
def lossFunction(games, weight_distance, weight_payoff_diff):
    '''
    Function to compute the loss.
    It is equal to weighted Euclidean distance of nash equilibria + weighted difference of payoffs resulting from the nash equilibria
    '''
    
    def enclosedLossFunction(nashEq_true, nashEq_pred):
        #Explicitly defined number of options because fuck, but at least it's a variable
        num_options = 3
        #Computing Euclidean distance of nash equilibria
        #L2Distance_nashEqs = K.sqrt(K.sum(K.square(nashEq_true - nashEq_pred)))
        
        #Computing the difference of payoffs resulting from the nash equilibria
        #payoffDifference = 0.1
        
        #Weighted sum of the two losses
        #loss = (weight_EqDistance * L2Distance_nashEqs) + (weight_payoffDiff * payoffDifference)
        
        #return loss
        # total = 0.0
        # final = 15000.0
        # val = 0.0
        # for i in range(0,len(nashEq_true)): #As big as needed
        #     val = 0.0
        #     for j in range(0, len(nashEq_true[i])): #2
        #         for k in range(0,len(nashEq_true[i][j])): #3
        #             #print(nashEq_pred[i][j][k])
        #             #print(nashEq_true[i][j][k])
        #             #print("****")
        #             sub = nashEq_true[i][j][k] - nashEq_pred[0][j][k]
        #             sub = pow(sub,2)
        #             val = val + sub
        # total = pow(val,(1/2))
        # if total < final:
        #     final = total

        # return final
        #loss_distance = K.min(K.sqrt(K.sum(K.square(nashEq_true - nashEq_pred))))
        #Calculate distance loss and find the index of the minimum distance label
        sq_diff = K.square(nashEq_true - nashEq_pred)
        distances = K.sqrt(K.sum(K.sum(sq_diff, axis=1), axis=1))
        index_dist = K.cast(K.argmin(distances), dtype='int32')
        loss_distance = K.gather(distances, index_dist)
        # loss_distance = distances[index_dist]
        
        #Get minimum distance nash equilibrium true
        nash_true_payoff = K.gather(nashEq_true, index_dist)
        payoff_true = get_payoff(games, nash_true_payoff, num_options)
        payoff_pred = get_payoff(games, nashEq_pred[0], num_options)

        #Find the difference between predicted and true payoffs
        payoff_sq_diff = K.square(payoff_true - payoff_pred)
        loss_payoff_distance = K.sqrt(K.sum(payoff_sq_diff))

        loss = weight_distance * loss_distance + weight_payoff_diff * loss_payoff_distance

        return loss
    
    return enclosedLossFunction


'''****************************'''
def GetTrainingDatafromCSV(filename):
    '''
    Function to read input CSV files
    '''
    
    df = pd.read_csv(filename, sep = ',', converters = {'Game' : ast.literal_eval, 'Nash' : ast.literal_eval})
    games = df.Game.tolist()
    equilibria = df.Nash.tolist()
    
    return games, equilibria


'''****************************'''
'''Load the main function when this module is directly executed'''
if __name__ == '__main__':
    main()

