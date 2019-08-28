#Headers
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import nashpy as nash
import random
from NashNet import *

#********************************
'''Global variables'''

#Redifining the number of examples
NUMBER_OF_EXAMPLES = 20

#Redefining the dataset file names
TEST_GAMES_FILES = ['Test_Games.npy']
TEST_EQUILIBRIA_FILES = ['Test_Equilibria.npy']

#Deciding to normalize the test data before computing the loss
NORMALIZE_INPUT_DATA = False


#********************************
def readTestData():
    '''
    Function to read the test data set aside during the last training session
    '''
    
    #Create arrays with predefined sizes
    sampleGames = np.zeros((0, PLAYER_NUMBER, PURE_STRATEGIES_PER_PLAYER, PURE_STRATEGIES_PER_PLAYER))
    sampleEquilibria = np.zeros((0, MAXIMUM_EQUILIBRIA_PER_GAME, PLAYER_NUMBER, PURE_STRATEGIES_PER_PLAYER))
    
    #Set where to look for the dataset files
    directory = './Reports/Saved_Test_Data/'
    if not os.path.isdir(directory):
        print('\n\nError: The directory for the test data does not exist.\n\n')
        exit()
    
    #Read the dataset files in npy format
    for gamesDataset, equilibriaDataset in zip(TEST_GAMES_FILES, TEST_EQUILIBRIA_FILES):
        sampleGames_temp, sampleEquilibria_temp = GetTrainingDataFromNPY(directory + gamesDataset, directory + equilibriaDataset)
        sampleGames = np.append(sampleGames, sampleGames_temp, axis = 0)
        sampleEquilibria = np.append(sampleEquilibria, sampleEquilibria_temp, axis = 0)
    
    return sampleGames, sampleEquilibria

#********************************
#Read datasets
testSamples, testEqs = readTestData()

#Shuffle the dataset arrays
testSamples, testEqs = unisonShuffle(testSamples, testEqs)

#Normalize the games
if NORMALIZE_INPUT_DATA:
    testSamples = (testSamples - np.reshape(np.min(testSamples, axis = (1, 2, 3)), (testSamples.shape[0], 1, 1, 1))) / np.reshape(np.max(testSamples, axis = (1, 2, 3)) - np.min(testSamples, axis = (1, 2, 3)), (testSamples.shape[0], 1, 1, 1))

#Loading the neural network model
with open('./Model/' + SAVED_MODEL_ARCHITECTURE_FILE + '.json') as json_file:
    json_config = json_file.read()
nn_model = keras.models.model_from_json(json_config)
nn_model.load_weights('./Model/' + SAVED_MODEL_WEIGHTS_FILE + '.h5')

#Print some examples of predictions
printExamples(NUMBER_OF_EXAMPLES, testSamples, testEqs, nn_model)
