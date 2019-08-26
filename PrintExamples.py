#Headers
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import nashpy as nash
import random
from NashNet import *

#Redifining the number of examples
NUMBER_OF_EXAMPLES = 20
NN_MODEL_FILE = './savedModel.hdf5'

#Read datasets
sampleGames, sampleEquilibria = readDatasets()

#Shuffle the dataset arrays
sampleGames, sampleEquilibria = unisonShuffle(sampleGames, sampleEquilibria)

#Extract the test data
testSamples = sampleGames[NUMBER_OF_TRAINING_SAMPLES : NUMBER_OF_TRAINING_SAMPLES + NUMBER_OF_TESTS_SAMPLES]
testEqs = sampleEquilibria[NUMBER_OF_TRAINING_SAMPLES : NUMBER_OF_TRAINING_SAMPLES + NUMBER_OF_TESTS_SAMPLES]

#Normalize the games
testSamples = (testSamples - np.reshape(np.min(testSamples, axis = (1, 2, 3)), (testSamples.shape[0], 1, 1, 1))) / np.reshape(np.max(testSamples, axis = (1, 2, 3)) - np.min(testSamples, axis = (1, 2, 3)), (testSamples.shape[0], 1, 1, 1))

#Loading the neural network model
with open(SAVED_MODEL_ARCHITECTURE_FILE + '.json') as json_file:
    json_config = json_file.read()
nn_model = keras.models.model_from_json(json_config)
nn_model.load_weights(SAVED_MODEL_WEIGHTS_FILE + '.h5')

#Print some examples of predictions
printExamples(NUMBER_OF_EXAMPLES, testSamples, testEqs, nn_model)
