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
# Initialize all global variables as None
PLAYER_NUMBER = None
PURE_STRATEGIES_PER_PLAYER = None
MAXIMUM_EQUILIBRIA_PER_GAME = None

#Loss Function
WEIGHT_EQUILIBRIA_DISTANCE = None
WEIGHT_PAYOFF_DIFFERENCE = None

#Neural Network Training
LEARNING_RATE = None
MOMENTUM = None
BATCH_SIZE = None
EPOCHS = None

#Training Strategy
NUMBER_OF_TRAINING_SAMPLES = None
NUMBER_OF_TESTS_SAMPLES = None
VALIDATION_SPLIT = None

#Dataset and Output
DATASET_GAMES_FILES = None
DATASET_EQUILIBRIA_FILES = None
NORMALIZE_INPUT_DATA = None
NUMBER_OF_EXAMPLES = None

#File Names
SAVED_MODEL_ARCHITECTURE_FILE = None
SAVED_MODEL_WEIGHTS_FILE = None
TRAINING_HISTORY_FILE = None
TEST_RESULTS_FILE = None
EXAMPLES_PRINT_FILE = None
SAVED_TEST_GAMES_FILE = None
SAVED_TEST_EQUILIBRIA_FILE = None

#Number of examples to print when running print examples
NUMBER_OF_PRINT_EXAMPLES = None

#Redefining the dataset file names
TEST_GAMES_FILES = None
TEST_EQUILIBRIA_FILES = None

def load_cfg(config_path):
    print("IN LOAD CFG")
    # D is for default
    d = "DEFAULT"

    # Create configparser object
    config = configparser.ConfigParser()

    # Success stores list of files successfully parsed. Throw error if empty
    success = config.read(config_path)
    d = 'DEFAULT' #Shortcut, for default config parser section

    # Check and make sure that a configuration file was read
    if len(success) <= 0:
        raise Exception("Fatal Error: No configuration file \'"+config_path+"\' found.")

    # Initialize all global variables as None
    global PLAYER_NUMBER
    global PURE_STRATEGIES_PER_PLAYER
    global MAXIMUM_EQUILIBRIA_PER_GAME
    
    #Loss Function
    global WEIGHT_EQUILIBRIA_DISTANCE
    global WEIGHT_PAYOFF_DIFFERENCE

    #Neural Network Training
    global LEARNING_RATE
    global MOMENTUM
    global BATCH_SIZE
    global EPOCHS

    #Training Strategy
    global NUMBER_OF_TRAINING_SAMPLES
    global NUMBER_OF_TESTS_SAMPLES
    global VALIDATION_SPLIT

    #Dataset and Output
    global DATASET_GAMES_FILES
    global DATASET_EQUILIBRIA_FILES
    global NORMALIZE_INPUT_DATA
    global NUMBER_OF_EXAMPLES

    #File Names
    global SAVED_MODEL_ARCHITECTURE_FILE
    global SAVED_MODEL_WEIGHTS_FILE
    global TRAINING_HISTORY_FILE
    global TEST_RESULTS_FILE
    global EXAMPLES_PRINT_FILE
    global SAVED_TEST_GAMES_FILE
    global SAVED_TEST_EQUILIBRIA_FILE

    # Number of examples to print when running print examples 
    global NUMBER_OF_PRINT_EXAMPLES

    # Test games to print
    global TEST_GAMES_FILES
    global TEST_EQUILIBRIA_FILES

    # Set values from config
    PLAYER_NUMBER = config.getint(d,"PLAYER_NUMBER")
    PURE_STRATEGIES_PER_PLAYER = config.getint(d,"PURE_STRATEGIES_PER_PLAYER")
    MAXIMUM_EQUILIBRIA_PER_GAME = config.getint(d,"MAXIMUM_EQUILIBRIA_PER_GAME")

    #Loss Function
    WEIGHT_EQUILIBRIA_DISTANCE = config.getfloat(d,"WEIGHT_EQUILIBRIA_DISTANCE")
    WEIGHT_PAYOFF_DIFFERENCE = config.getfloat(d,"WEIGHT_PAYOFF_DIFFERENCE")

    #Neural Network Training
    LEARNING_RATE = config.getfloat(d, "LEARNING_RATE")
    MOMENTUM = config.getfloat(d, "MOMENTUM")
    BATCH_SIZE = config.getint(d, "BATCH_SIZE")
    EPOCHS = config.getint(d, "EPOCHS")

    #Training Strategy
    NUMBER_OF_TRAINING_SAMPLES = config.getint(d, "NUMBER_OF_TRAINING_SAMPLES")
    NUMBER_OF_TESTS_SAMPLES = config.getint(d, "NUMBER_OF_TESTS_SAMPLES")
    VALIDATION_SPLIT = config.getfloat(d, "VALIDATION_SPLIT")

    #Dataset and Output
    DATASET_GAMES_FILES = str_to_list(config.get(d, "DATASET_GAMES_FILES"))
    DATASET_EQUILIBRIA_FILES = str_to_list(config.get(d, "DATASET_EQUILIBRIA_FILES"))
    NORMALIZE_INPUT_DATA = config.getboolean(d, "NORMALIZE_INPUT_DATA")
    NUMBER_OF_EXAMPLES = config.getint(d, "NUMBER_OF_EXAMPLES")

    #File Names
    SAVED_MODEL_ARCHITECTURE_FILE = config.get(d, "SAVED_MODEL_ARCHITECTURE_FILE")
    SAVED_MODEL_WEIGHTS_FILE = config.get(d, "SAVED_MODEL_WEIGHTS_FILE")
    TRAINING_HISTORY_FILE = config.get(d, "TRAINING_HISTORY_FILE")
    TEST_RESULTS_FILE = config.get(d, "TEST_RESULTS_FILE")
    EXAMPLES_PRINT_FILE = config.get(d, "EXAMPLES_PRINT_FILE")
    SAVED_TEST_GAMES_FILE = config.get(d, "SAVED_TEST_GAMES_FILE")
    SAVED_TEST_EQUILIBRIA_FILE = config.get(d, "SAVED_TEST_EQUILIBRIA_FILE")

    # Number of examples to print when running
    NUMBER_OF_PRINT_EXAMPLES = config.getint(d, "NUMBER_OF_PRINT_EXAMPLES")

    # Redefining the dataset file names
    TEST_GAMES_FILES = str_to_list(config.get(d, "TEST_GAMES_FILES"))
    TEST_EQUILIBRIA_FILES = str_to_list(config.get(d, "TEST_EQUILIBRIA_FILES"))

    
load_cfg("./Configs/example.cfg")

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
printExamples(NUMBER_OF_PRINT_EXAMPLES, testSamples, testEqs, nn_model)
