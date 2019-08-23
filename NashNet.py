#Headers
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import ast
import pandas as pd
import nashpy as nash
import random
import matplotlib.pyplot as plt


#********************************
'''Global variables'''
#Game Settings
PLAYER_NUMBER = 2
PURE_STRATEGIES_PER_PLAYER = 3
MAXIMUM_EQUILIBRIA_PER_GAME = 10

#Loss Function
WEIGHT_EQUILIBRIA_DISTANCE = 1
WEIGHT_PAYOFF_DIFFERENCE = 0.5

#Neural Network Training
LEARNING_RATE = 0.01
BATCH_SIZE = 128
EPOCHS = 500

#Training Strategy
NUMBER_OF_TRAINING_SAMPLES = 290000
NUMBER_OF_TESTS_SAMPLES = 10000
VALIDATION_SPLIT = 0.2

#Dataset and Output
DATASET_GAMES_FILES = ['Dataset-DiscardedPure-1_Games_2P-3x3_1M.npy.npy'] #['Dataset-1_Games_2P-3x3_1M.npy', 'Dataset-2_Games_2P-3x3_1M.npy']
DATASET_EQUILIBRIA_FILES = ['Dataset-DiscardedPure-1_Equilibria_2P-3x3_1M.npy'] #['Dataset-1_Equilibria_2P-3x3_1M.npy', 'Dataset-2_Equilibria_2P-3x3_1M.npy']
SAVED_MODEL_ARCHITECTURE_FILE = 'modelArchitecture'
SAVED_MODEL_WEIGHTS_FILE = 'modelWeights'
TRAINING_HISTORY_FILE = 'training_history.csv'
TEST_RESULTS_FILE = 'test_results.csv'
EXAMPLES_PRINT_FILE = 'printed_examples.txt'
NUMBER_OF_EXAMPLES = 2

#********************************
def main():
    '''
    Main function
    '''
    
    #Read the dataset
    #Indexing:
    #trainingSamples: [sample #] [player #] [row # of the game] [column # of the game]
    #trainingEqs: [sample #] [equilibrium #] [player #] [pure strategy #]
    sampleGames, sampleEquilibria = readDatasets()

    #Extract the training and test data
    trainingSamples = sampleGames[ : NUMBER_OF_TRAINING_SAMPLES]
    trainingEqs = sampleEquilibria[ : NUMBER_OF_TRAINING_SAMPLES]
    testSamples = sampleGames[NUMBER_OF_TRAINING_SAMPLES : NUMBER_OF_TRAINING_SAMPLES + NUMBER_OF_TESTS_SAMPLES]
    testEqs = sampleEquilibria[NUMBER_OF_TRAINING_SAMPLES : NUMBER_OF_TRAINING_SAMPLES + NUMBER_OF_TESTS_SAMPLES]
    
    #Normalize the games
    trainingSamples = (trainingSamples - np.reshape(np.min(trainingSamples, axis = (1, 2, 3)), (trainingSamples.shape[0], 1, 1, 1))) / np.reshape(np.max(trainingSamples, axis = (1, 2, 3)) - np.min(trainingSamples, axis = (1, 2, 3)), (trainingSamples.shape[0], 1, 1, 1))
    testSamples = (testSamples - np.reshape(np.min(testSamples, axis = (1, 2, 3)), (testSamples.shape[0], 1, 1, 1))) / np.reshape(np.max(testSamples, axis = (1, 2, 3)) - np.min(testSamples, axis = (1, 2, 3)), (testSamples.shape[0], 1, 1, 1))
#     trainingSamples = (trainingSamples - np.reshape(np.average(trainingSamples, axis = (1, 2, 3)), (trainingSamples.shape[0], 1, 1, 1))) / np.reshape(np.std(trainingSamples, axis = (1, 2, 3)), (trainingSamples.shape[0], 1, 1, 1))
#     testSamples = (testSamples - tf.reshape(np.average(testSamples, axis = (1, 2, 3)), (testSamples.shape[0], 1, 1, 1))) / np.reshape(np.std(testSamples, axis = (1, 2, 3)), (testSamples.shape[0], 1, 1, 1))
    
    #Constructing the neural network
    #Input layer

    nn_Input = layers.Input(shape = tuple((PLAYER_NUMBER,) + tuple(PURE_STRATEGIES_PER_PLAYER for _ in range(PLAYER_NUMBER))))
    flattenedInput = layers.Flatten(input_shape = tuple((PLAYER_NUMBER,) + tuple(PURE_STRATEGIES_PER_PLAYER for _ in range(PLAYER_NUMBER))))(nn_Input)
    
    #Fully-connected layers
    layer1 = layers.Dense(20, activation = 'relu')(flattenedInput)
    layer2 = layers.Dense(50, activation = 'relu')(layer1)
#     layer2do = layers.Dropout(0.2)(layer2)
    layer3 = layers.Dense(50, activation = 'relu')(layer2)
#     layer3do = layers.Dropout(0.2)(layer3)
    layer4 = layers.Dense(50, activation = 'relu')(layer3)
#     layer4do = layers.Dropout(0.2)(layer4)
    layer5 = layers.Dense(50, activation = 'relu')(layer4)
#     layer5do = layers.Dropout(0.2)(layer5)
    layer6 = layers.Dense(50, activation = 'relu')(layer5)
#     layer6do = layers.Dropout(0.2)(layer6)
    layer7 = layers.Dense(50, activation = 'relu')(layer6)
#     layer7do = layers.Dropout(0.2)(layer7)

#     layer7a = layers.Dense(50, activation = 'relu')(layer7)
#     layer7b = layers.Dense(50, activation = 'relu')(layer7a)
#     layer7c = layers.Dense(50, activation = 'relu')(layer7b)
#     layer7d = layers.Dense(50, activation = 'relu')(layer7c)
#     layer7e = layers.Dense(50, activation = 'relu')(layer7d)

    layer8 = layers.Dense(20, activation = 'relu')(layer7)
#     layer8do = layers.Dropout(0.2)(layer8)
    layer9 = layers.Dense(10, activation = 'relu')(layer8)
    
    lastLayer_player = [layers.Dense(PURE_STRATEGIES_PER_PLAYER)(layer9)]
    for _ in range(1, PLAYER_NUMBER):
        lastLayer_player.append(layers.Dense(PURE_STRATEGIES_PER_PLAYER)(layer9))

    #Softmax layers. Two for the two players
    softmax = [layers.Activation('softmax')(lastLayer_player[0])]
    for playerCounter in range(1, PLAYER_NUMBER):
        softmax.append(layers.Activation('softmax')(lastLayer_player[playerCounter]))
    
    #Output layer
    concatenated_output = layers.concatenate([softmax[pl] for pl in range(PLAYER_NUMBER)])
    replicated_output = layers.concatenate([concatenated_output for i in range(MAXIMUM_EQUILIBRIA_PER_GAME)])
    nn_Output = layers.Reshape((MAXIMUM_EQUILIBRIA_PER_GAME, PLAYER_NUMBER, PURE_STRATEGIES_PER_PLAYER))(replicated_output)
    
    #Defining the NN model
    nn_model = keras.Model(inputs = nn_Input, outputs = nn_Output)
    
    # Create the optimizer
    Optimizer = keras.optimizers.SGD(learning_rate = LEARNING_RATE)

    #Compiling the NN model
    nn_model.compile(loss = lossFunction(nn_Input, WEIGHT_EQUILIBRIA_DISTANCE, WEIGHT_PAYOFF_DIFFERENCE, PURE_STRATEGIES_PER_PLAYER), optimizer = Optimizer, 
                     metrics = [MSE, PayoffLoss_metric(nn_Input, WEIGHT_EQUILIBRIA_DISTANCE, WEIGHT_PAYOFF_DIFFERENCE, PURE_STRATEGIES_PER_PLAYER)])

    #Printing the summary of the constructed model
    print(nn_model.summary())
    
    #Train the NN model
    trainingHistory = nn_model.fit(trainingSamples, trainingEqs, validation_split = VALIDATION_SPLIT, epochs = EPOCHS, batch_size = BATCH_SIZE, shuffle = True)
    
    #Test the trained model
    evaluationResults = nn_model.evaluate(testSamples, testEqs, batch_size = 256)
    
    #Save the model
    with open(SAVED_MODEL_ARCHITECTURE_FILE + '.json', 'w') as json_file:
        json_file.write(nn_model.to_json())
    nn_model.save_weights(SAVED_MODEL_WEIGHTS_FILE + '.h5')
    
    #Write the loss and metric values during the training and test time
    pd.DataFrame(trainingHistory.history).to_csv(TRAINING_HISTORY_FILE)
    pd.DataFrame([nn_model.metrics_names, evaluationResults]).to_csv(TEST_RESULTS_FILE, index = False)
    
    #Print some examples of predictions
    printExamples(NUMBER_OF_EXAMPLES, testSamples, testEqs, nn_model)
    
#********************************
def MSE(nashEq_true, nashEq_pred):
    '''
    Function to compute the correct mean squared error (mse) as a metric during model training and testing
    '''
    
    return K.mean(K.min(K.mean(K.square(nashEq_true - nashEq_pred), axis = [2, 3]), axis = 1))

#********************************
def PayoffLoss_metric(game, weight_EqDiff, weight_payoffDiff, pureStrategies_perPlayer):
    '''
    Function to compute payoff loss as a metric during model training and testing
    '''
    
    def PayoffLoss(nashEq_true, nashEq_pred):
        #Computing the minimum of sum of squared error (SSE) of nash equilibria
        SSE_eq = K.sum(K.square(nashEq_true - nashEq_pred), axis = [2, 3])
        min_index = K.argmin(SSE_eq, axis = 1)
        
        #Computing the minimum of sum of sqaured error (SSE) of payoffs
        selected_trueNash = tf.gather_nd(nashEq_true, tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype = 'int32')), axis = 1))
        payoff_true = computePayoff(game, selected_trueNash, pureStrategies_perPlayer)
        payoff_pred = computePayoff(game, tf.gather(nashEq_pred, 0, axis = 1), pureStrategies_perPlayer)

        #Find the difference between the predicted and true payoffs
        payoff_sqDiff = K.square(payoff_true - payoff_pred)
        loss_payoffs_SSE = K.mean(K.sum(payoff_sqDiff, axis = 1))

        return loss_payoffs_SSE
     
    return PayoffLoss
    
#********************************
def lossFunction(game, weight_EqDiff, weight_payoffDiff, pureStrategies_perPlayer):
    '''
    Function to compute the loss.
    It is equal to weighted Euclidean distance of nash equilibria + weighted difference of payoffs resulting from the nash equilibria
    '''
    
    def enclosedLossFunction(nashEq_true, nashEq_pred):
        #Computing the minimum of sum of squared error (SSE) of nash equilibria
        SSE_eq = K.sum(K.square(nashEq_true - nashEq_pred), axis = [2, 3])
        min_index = K.argmin(SSE_eq, axis = 1)
        loss_Eqs_SSE = tf.gather_nd(SSE_eq, tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype = 'int32')), axis = 1))

        #Computing the minimum of sum of sqaured error (SSE) of payoffs
        selected_trueNash = tf.gather_nd(nashEq_true, tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype = 'int32')), axis = 1))
        payoff_true = computePayoff(game, selected_trueNash, pureStrategies_perPlayer)
        payoff_pred = computePayoff(game, tf.gather(nashEq_pred, 0, axis = 1), pureStrategies_perPlayer)
        
        #Find the difference between the predicted and true payoffs
        payoff_sqDiff = K.square(payoff_true - payoff_pred)
        loss_payoffs_SSE = K.sum(payoff_sqDiff, axis = 1) 
        
        #Weighted sum of the two losses
        loss = loss_Eqs_SSE
#         loss = loss_Eqs_SSE * tf.tanh(loss_payoffs_SSE)
#         loss = weight_EqDiff * loss_Eqs_SSE + weight_payoffDiff * loss_payoffs_SSE
#         loss = loss_payoffs_SSE
        
        loss = K.mean(loss)
        
        return loss
    
    return enclosedLossFunction

#********************************
def computePayoff(game, equilibrium, pureStrategies_perPlayer):
    '''
    Function to compute the payoff each player gets with the input equilibrium and the input game (2 player games).
    '''
    
    #Extract mix strategies of each player
    mixStrategies_p1 = tf.gather(equilibrium, 0, axis = 1)
    mixStrategies_p1 = K.reshape(mixStrategies_p1, (tf.shape(mixStrategies_p1)[0], pureStrategies_perPlayer, 1))
    mixStrategies_p2 = tf.gather(equilibrium, 1, axis = 1)
    mixStrategies_p2 = K.reshape(mixStrategies_p2, (tf.shape(mixStrategies_p2)[0], 1, pureStrategies_perPlayer))
    
    #Multiply them together to get the probability matrix
    probability_mat = mixStrategies_p1 * mixStrategies_p2
    
    #Adding a new dimension
    probability_mat = K.expand_dims(probability_mat, axis = 1)

    #Concatenate probability mat with itself to get a tensor with shape (2, pureStrategies_perPlayer, pureStrategies_perPlayer)
    probability_mat = K.concatenate([probability_mat, probability_mat], axis = 1)

    #Multiply the probability matrix by the game (payoffs) to get the expected payoffs for each player
    expectedPayoff_mat = game * probability_mat

    #Sum the expected payoff matrix for each player (eg: (Batch_Size, 2,3,3)->(Batch_Size, 2,1))
    payoffs = K.sum(expectedPayoff_mat, axis = [2, 3])

    return payoffs


#********************************
def computePayoff_np(game, equilibrium, pureStrategies_perPlayer, playerNumber):
    '''
    Function to compute the payoff each player gets with the input equilibrium and the input game (games with more than 2 players).
    '''    
    
    #Extract mix strategies of each player
    mixStrategies_perPlayer = [tf.gather(equilibrium, 0, axis = 1) for pl in range(playerNumber)]
    playerProbShape_grid = tuple(np.ones(playerNumber) + np.identity(playerNumber) * (pureStrategies_perPlayer - 1))
    playerProbShape = [tuple((tf.shape(mixStrategies_perPlayer[0])[0], ) + tuple(playerProbShape_grid[pl])) for pl in range(playerNumber)]
    mixStrategies_perPlayer = [K.reshape(mixStrategies_perPlayer[pl], playerProbShape[pl]) for pl in range(playerNumber)]

    #Multiply them together to get the probability matrix
    probability_mat = mixStrategies_perPlayer[0]
    for pl in range(1, playerNumber):
        probability_mat *= mixStrategies_perPlayer[pl]

    #Adding a new dimension
    probability_mat = K.expand_dims(probability_mat, axis = 1)

    #Concatenate probability mat with itself to get a tensor with shape (2, pureStrategies_perPlayer, pureStrategies_perPlayer)
    probability_mat = K.concatenate([probability_mat, probability_mat], axis = 1)

    #Multiply the probability matrix by the game (payoffs) to get the expected payoffs for each player
    expectedPayoff_mat = game * probability_mat

    #Sum the expected payoff matrix for each player (eg: (Batch_Size, 2,3,3)->(Batch_Size, 2,1))
    payoffs = K.sum(expectedPayoff_mat, axis = [2, 3])

    return payoffs

#********************************
def GetTrainingDatafromCSV(filename):
    '''
    Function to read input CSV files
    '''
    
    df = pd.read_csv(filename, sep = ',', converters = {'Game' : ast.literal_eval, 'Nash' : ast.literal_eval})
    games = np.array(df.Game.tolist())
    equilibria = np.array(df.Nash.tolist())
    
    return games, equilibria

#********************************
def GetTrainingDataFromNPY(data_file, labels_file):
    '''
    Function to load dataset from two .npy files
    '''
    
    data = np.load(data_file)
    labels = np.load(labels_file)

    return data, labels

#********************************
def readDatasets():
    '''
    Function to read datasets
    '''
    
    #Indexing:
    #trainingSamples: [sample #] [player #] [row # of the game] [column # of the game]
    #trainingEqs: [sample #] [equilibrium #] [player #] [pure strategy #]
    sampleGames = np.zeros((0, PLAYER_NUMBER, PURE_STRATEGIES_PER_PLAYER, PURE_STRATEGIES_PER_PLAYER))
    sampleEquilibria = np.zeros((0, MAXIMUM_EQUILIBRIA_PER_GAME, PLAYER_NUMBER, PURE_STRATEGIES_PER_PLAYER))
    
    for gamesDataset, equilibriaDataset in zip(DATASET_GAMES_FILES, DATASET_EQUILIBRIA_FILES):
        sampleGames_temp, sampleEquilibria_temp = GetTrainingDataFromNPY('./Datasets/' + gamesDataset, './Datasets/' + equilibriaDataset)
        sampleGames = np.append(sampleGames, sampleGames_temp, axis = 0)
        sampleEquilibria = np.append(sampleEquilibria, sampleEquilibria_temp, axis = 0)
    
    return sampleGames, sampleEquilibria

#********************************
def generate_nash(game):
    '''
    Function to compute Nash equilibrium  based on the classical methods
    '''
    
    equilibrium = nash.Game(game[0], game[1])
    
    nash_support_enumeration = []
    nash_lemke_howson_enumeration = []
    nash_vertex_enumeration = []

    for eq in equilibrium.support_enumeration():
        nash_support_enumeration.append(eq)
        
    for eq in equilibrium.lemke_howson_enumeration():
        nash_lemke_howson_enumeration.append(eq)
        
    for eq in equilibrium.vertex_enumeration():
        nash_vertex_enumeration.append(eq)
        
    return nash_support_enumeration, nash_lemke_howson_enumeration, nash_vertex_enumeration

#********************************
def printExamples(numberOfExamples, testSamples, testEqs, nn_model):
    '''
    Function to make some illustrative predictions and print them
    '''
    
    #Check the rquested number of examples is feasible
    if numberOfExamples > testSamples.shape[0]:
        print("\n\nNumber of example predictions more than the number of test samples\n")
        exit()
    
    #Fetching an example game from the test set
    randomExample = random.randint(0, testSamples.shape[0] - numberOfExamples)
    exampleGame = testSamples[randomExample : randomExample + numberOfExamples]
    nash_true = testEqs[randomExample : randomExample + numberOfExamples]
    nash_true = nash_true.astype('float32')
    
    #Predicting a Nash equilibrium for the example game
    nash_predicted = nn_model.predict(exampleGame)

    #Open file for writing the results into
    printFile = open(EXAMPLES_PRINT_FILE, "w")
    
    for exampleCounter in range(numberOfExamples):
        #Computing the loss
        lossFunction_instance = lossFunction(exampleGame[exampleCounter], WEIGHT_EQUILIBRIA_DISTANCE, WEIGHT_PAYOFF_DIFFERENCE, PURE_STRATEGIES_PER_PLAYER)
        loss = lossFunction_instance(np.expand_dims(nash_true[exampleCounter], axis = 0), np.expand_dims(nash_predicted[exampleCounter], axis = 0))

        #Compute the Nash equilibrium for the current game to get only distinctive equilibria
        distinctive_NashEquilibria, _, _ = generate_nash(exampleGame[exampleCounter])
        
        #Printing the results for the example game
        listOfTrueEquilibria = [distinctive_NashEquilibria[i] for i in range(len(distinctive_NashEquilibria))]
        printString = ("\n______________\nExample {}:\nTrue: \n" + ("{}\n" * len(distinctive_NashEquilibria)) + "\nPredicted: \n{}\n\nLoss: {}\n\n") \
              .format(* ([exampleCounter + 1] + listOfTrueEquilibria + list([nash_predicted[exampleCounter, 0]]) + [K.get_value(loss)])).replace("array", "")
        print(printString)
        
        #Write the string to the file
        printFile.write(printString)
    
    printFile.close()
    
    return 

#********************************
'''Load the main function when this module is directly executed'''
if __name__ == '__main__':
    main()

