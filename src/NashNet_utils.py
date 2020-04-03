import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import math
import pandas as pd
from sklearn import cluster
import os
from Loss import *


# ********************************
def unisonShuffle(a, b):
    """
    Function to shuffle two numpy arrays in unison
    """

    assert a.shape[0] == b.shape[0]

    p = np.random.permutation(a.shape[0])

    return a[p], b[p]


# ********************************
def save_training_history(trainingHistory, trainingHistory_file):
    """
    Function to save the training history in a file
    """

    # Save training history
    trainingHistory_dataFrame = pd.DataFrame(trainingHistory.history)
    trainingHistory_dataFrame.index += 1
    trainingHistory_dataFrame.to_csv('./Reports/' + trainingHistory_file)


# ********************************
def saveDataSplit(files_list, save_list_file, data_split_folder, num_players, num_strategies):
    """
    Function to save list of test/validation/training data to reuse later
    """

    address = './Datasets/' + str(num_players) + 'P/' + str(num_strategies[0])
    for strategy in num_strategies[1:]:
        address += 'x' + str(strategy)

    assert os.path.exists(address), 'The path ' + address + ' does not exist'

    address = os.path.join(address, data_split_folder)
    if not os.path.exists(address):
        os.mkdir(address)

    save_list_file += '.npy'

    np.save(os.path.join(address, save_list_file), files_list)


# ********************************
def loadDataSplit(saved_files_list, data_split_folder, num_players, num_strategies):
    """
    Function to load test/validation/training data
    """

    address = './Datasets/' + str(num_players) + 'P/' + str(num_strategies[0])
    for strategy in num_strategies[1:]:
        address += 'x' + str(strategy)
    address = os.path.join(address, data_split_folder)

    saved_files_list += '.npy'

    files_list = np.load(os.path.join(address, saved_files_list))

    return files_list


# ********************************
def saveModel(model, model_architecture_file, model_weights_file):
    """
    Function to save the trained model
    """

    # Save model architecture
    with open('./Model/' + model_architecture_file + '.json', 'w') as json_file:
        json_file.write(model.to_json())

    # Save model weights
    model.save_weights('./Model/' + model_weights_file + '.h5')


# ********************************
def printExamples(numberOfExamples, test_data_generator, nn_model, examples_print_file, pureStrategies_per_player,
                  lossType, payoffLoss_type, num_players, enable_hydra, cluster_examples, print_to_terminal,
                  payoffToEq_weight=None):
    """
    Function to make some illustrative predictions and print them
    """

    # Get the number of batches in the test data generator
    number_of_batches = test_data_generator.__len__()

    # Get an initial batch
    initial_game_batch, initial_eq_batch = test_data_generator.__getitem__(0)

    # Check the requested number of examples is feasible
    if numberOfExamples > (initial_game_batch.shape[0] * number_of_batches):
        print("\n\nNumber of example predictions more than the number of test samples\n")
        exit()
    elif numberOfExamples == 0:
        return

    # Fetching example games from the test set
    random_batches = np.random.permutation(number_of_batches)
    remaining_examples = numberOfExamples
    batch_counter = 0
    exampleGame = np.zeros((numberOfExamples,) + initial_game_batch.shape[1:], dtype='float32')
    nash_true = np.zeros((numberOfExamples,) + initial_eq_batch.shape[1:],  dtype='float32')

    while remaining_examples > 0:
        # Get a random batch
        games_batch, eq_batch = test_data_generator.__getitem__(random_batches[batch_counter])
        batch_counter += 1

        # Accumulate the examples
        start_index = numberOfExamples - remaining_examples
        batch_size = games_batch.shape[0]
        if remaining_examples > batch_size:
            exampleGame[start_index: start_index + batch_size] = games_batch.astype('float32')
            nash_true[start_index: start_index + batch_size] = eq_batch.astype('float32')
        else:
            exampleGame[start_index:] = games_batch[0: remaining_examples].astype('float32')
            nash_true[start_index:] = eq_batch[0: remaining_examples].astype('float32')

        # Update the number of remaining examples to get
        remaining_examples -= batch_size

    # Predicting a Nash equilibrium for the example game
    nash_predicted = nn_model.predict(exampleGame).astype('float32')

    # Determine which loss function to use
    lossFunction, payoffLoss_function, computePayoff_function = chooseLossFunction(lossType, payoffLoss_type,
                                                                                   num_players, enable_hydra)

    # Open file for writing the results into
    printFile = open('./Reports/' + examples_print_file, "w")

    for exampleCounter in range(numberOfExamples):
        # Computing the loss
        lossFunction_instance = lossFunction(np.expand_dims(exampleGame[exampleCounter], axis=0), payoffLoss_function, payoffToEq_weight,
                                             pureStrategies_per_player, computePayoff_function, num_players)
        loss = lossFunction_instance(np.expand_dims(nash_true[exampleCounter], axis=0),
                                     np.expand_dims(nash_predicted[exampleCounter], axis=0))

        # Cluster the Nash equilibria for the current game to get only distinctive equilibria
        listOfTrueEquilibria = np.where(np.isnan(nash_true[exampleCounter][0]), np.zeros_like(nash_true[exampleCounter]),
                 nash_true[exampleCounter])
        listOfTrueEquilibria = clustering(
            np.reshape(listOfTrueEquilibria, (nash_true.shape[1], num_players * max(pureStrategies_per_player))),
            num_players, max(pureStrategies_per_player))

        # Replace zero on the redundant values before doing any possible clustering
        nash_predicted[exampleCounter] = np.where(np.isnan(nash_true[exampleCounter][0]), np.zeros_like(nash_predicted[exampleCounter]), nash_predicted[exampleCounter])

        # If enabled, cluster the predicted Nash equilibria
        if cluster_examples:
            predictedEq = clustering(np.reshape(nash_predicted[exampleCounter],
                                                (nash_predicted.shape[1], num_players * max(pureStrategies_per_player))),
                                     num_players, max(pureStrategies_per_player))
        else:
            predictedEq = nash_predicted[exampleCounter]

        # Convert the numpy arrays to nested lists
        true_equilibria = [np.round(eq.astype(np.float), decimals=4).tolist() for eq in listOfTrueEquilibria]
        predicted_equilibria = np.round(predictedEq.astype(np.float), decimals=4).tolist()
        current_example_game = [np.round(player_game.astype(np.float), decimals=4).tolist() for player_game in exampleGame[exampleCounter]]

        # Remove redundant elements from equilibrium arrays
        for eq in range(len(true_equilibria)):
            true_equilibria[eq] = [true_equilibria[eq][pl][: pureStrategies_per_player[pl]] for pl in range(num_players)]

        for eq in range(len(predicted_equilibria)):
            predicted_equilibria[eq] = [predicted_equilibria[eq][pl][: pureStrategies_per_player[pl]] for pl in range(num_players)]

        printString = ("\n______________\nExample {}:\nGame:\n" + "{}\n" * len(current_example_game) + "\nTrue:\n" + "{}\n" * len(true_equilibria) + "\n\nPredicted: \n" + "{}\n" * len(predicted_equilibria) + "\n\nLoss: {:.4f}\n")\
            .format(*([exampleCounter + 1] + current_example_game + true_equilibria + predicted_equilibria + [K.get_value(loss)]))

        if print_to_terminal:
            print(printString)

        # Write the string to the file
        printFile.write(printString)

    printFile.close()


# ********************************
class TrainingCallback(tf.keras.callbacks.Callback):
    """
    Class to save model and change learning rate during the training
    """
    def __init__(self, initial_lr, num_cycles, max_epochs, save_dir, save_name, save_interim_weights):
        # Learning Rate Scheduler Variables
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.num_cycles = num_cycles

        # Checkpoint and saving variables
        self.save_dir = save_dir
        self.save_name = save_name
        self.save_interim_weights = save_interim_weights

    #     def on_train_begin(self, logs=None):
    #         return

    #     def on_train_end(self, logs=None):
    #         return

    def on_epoch_begin(self, epoch, logs=None):
        # Calculate new learning rate
        new_lr = self.initial_lr / 2 * (math.cos(
            math.pi * ((epoch) % (self.max_epochs / self.num_cycles)) / (self.max_epochs / self.num_cycles)) + 1)

        # Update learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

    def on_epoch_end(self, epoch, logs=None):
        # If model is at minima (before learning rate goes up again), save the model
        if (epoch % (self.max_epochs / self.num_cycles) == 0) and (epoch != 0) and self.save_interim_weights:
            # Get snapshot number
            snapshot_num = int(epoch / int(self.max_epochs / self.num_cycles))

            # Save the weights
            self.model.save_weights(
                self.save_dir.rstrip('/') + "/" + self.save_name + '_snapshot' + str(snapshot_num) + '_epoch' + str(
                    epoch) + '.h5')

#     def on_batch_begin(self, batch, logs={}):
#         return

#     def on_batch_end(self, batch, logs={}):
#         return


# ********************************
def clustering(pred, num_players, max_pureStrategies):
    """
    Function to cluster the predicted Nash equilibria.
    """

    # Run the clustering algorithm
    clustering = cluster.DBSCAN(eps=0.1, min_samples=1, metric='l2').fit(pred)

    # Find the number of clusters
    clusterNumber = np.max(clustering.labels_) + 1

    # Return the cluster centers
    return np.reshape(np.array(
        [np.mean(clustering.components_[np.where(clustering.labels_ == clusterCounter)], axis=0) for clusterCounter in
         range(clusterNumber)]), (clusterNumber, num_players, max_pureStrategies))
