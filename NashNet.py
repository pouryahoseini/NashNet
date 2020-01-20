import os, configparser, ast
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from NashNet_utils import *
import logging


# ********************************
class NashNet:
    """
    The class to hold the NashNet data and methods to train and evaluate that.
    To create a NashNet class, a configuration file should be provided: "NashNet(configFile, [config section])".
    """

    # ******
    def __init__(self, configFile, configSection="DEFAULT", initial_modelWeights=None):
        """
        Constructor function
        """

        # Initialize variables
        self.trained_model_loaded = False
        self.test_games = self.test_equilibria = np.array([])

        # Set Tensorflow's verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # WARN
        logging.getLogger('tensorflow').setLevel(logging.WARN)

        # Load config
        self.load_config(configFile, configSection=configSection)

        # Build the neural network model
        if not self.cfg["enable_hydra"]:
            self.model = build_model(num_players=self.cfg["num_players"],
                                     pure_strategies_per_player=self.cfg["num_strategies"],
                                     max_equilibria=self.cfg["max_equilibria"],
                                     optimizer=tf.keras.optimizers.SGD(
                                         learning_rate=self.cfg["initial_learning_rate"],
                                         momentum=self.cfg["momentum"],
                                         nesterov=self.cfg["nesterov"]),
                                     lossType=self.cfg["loss_type"],
                                     payoffLoss_type=self.cfg["payoff_loss_type"],
                                     enable_batchNormalization=self.cfg["batch_normalization"],
                                     payoffToEq_weight=self.cfg["payoff_to_equilibrium_weight"],
                                     compute_epsilon=self.cfg["compute_epsilon"]
                                     )
        else:
            self.model = build_hydra_model(num_players=self.cfg["num_players"],
                                           pure_strategies_per_player=self.cfg["num_strategies"],
                                           max_equilibria=self.cfg["max_equilibria"],
                                           optimizer=tf.keras.optimizers.SGD(
                                               learning_rate=self.cfg["initial_learning_rate"],
                                               momentum=self.cfg["momentum"],
                                               nesterov=self.cfg["nesterov"]),
                                           lossType=self.cfg["loss_type"],
                                           payoffLoss_type=self.cfg["payoff_loss_type"],
                                           enable_batchNormalization=self.cfg["batch_normalization"],
                                           hydra_shape=self.cfg["hydra_physique"],
                                           payoffToEq_weight=self.cfg["payoff_to_equilibrium_weight"],
                                           compute_epsilon=self.cfg["compute_epsilon"]
                                           )

        # Load initial model weights if any one is provided in the config file or in the constructor arguments
        self.weights_initialized = False
        if initial_modelWeights:
            self.model.load_weights(initial_modelWeights)
            self.weights_initialized = True
        elif self.cfg["initial_model_weights"]:
            self.model.load_weights(self.cfg["initial_model_weights"])
            self.weights_initialized = True

    # ******
    def train(self):
        """
        Function to train NashNet.
        """

        # Load the training data
        training_games, training_equilibria, self.test_games, self.test_equilibria = self.load_datasets()

        # Write the test data
        if self.cfg["rewrite_saved_test_data_if_model_weights_given"] or (not self.weights_initialized):
            saveTestData(self.test_games, self.test_equilibria, self.cfg["num_players"], self.cfg["num_strategies"])

        # Print the summary of the model
        print(self.model.summary())

        # Create the list of callbacks
        callbacks_list = [TrainingCallback(initial_lr=self.cfg["initial_learning_rate"],
                                           num_cycles=self.cfg["num_cycles"],
                                           max_epochs=self.cfg["epochs"],
                                           save_dir='./Model/Interim/',
                                           save_name="interimWeight"),
                          keras.callbacks.ModelCheckpoint(filepath='./Model/' + self.cfg[
                              "model_best_weights_file"] + '.h5',
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          mode='min',
                                                          save_freq='epoch')]

        if self.cfg["compute_epsilon"]:
            callbacks_list += [EpsilonCallback()]

        # Train the model
        trainingHistory = self.model.fit(training_games, training_equilibria,
                                         validation_split=self.cfg["validation_split"],
                                         epochs=self.cfg["epochs"],
                                         batch_size=self.cfg["batch_size"],
                                         shuffle=True,
                                         callbacks=callbacks_list
                                         )

        # Save the model
        saveModel(self.model, self.cfg["model_architecture_file"], self.cfg["model_weights_file"])

        # Write the loss and metric values during the training and test time
        save_training_history(trainingHistory, self.cfg["training_history_file"])

        # Indicate trained model is ready
        self.trained_model_loaded = True

    # ******
    def evaluate(self, num_to_print=None):
        """
        Function to evaluate NashNet.
        """

        # Load model weights if the model is not loaded yet
        if not self.trained_model_loaded:
            self.model.load_weights(os.path.join('./Model/' + self.cfg["model_weights_file"] + '.h5'))
            self.trained_model_loaded = True

        # If no num_to_print is provided, default to setting in cfg
        if not num_to_print:
            num_to_print = self.cfg["examples_to_print"]

        # Load test data if not already created
        if not (self.test_games.size and self.test_equilibria.size):
            self.test_games, self.test_equilibria = loadTestData(self.cfg["test_games_file"],
                                                                 self.cfg["test_equilibria_file"],
                                                                 self.cfg["max_equilibria"], self.cfg["num_players"],
                                                                 self.cfg["num_strategies"])

        # Create the list of callbacks
        callbacks_list = []
        if self.cfg["compute_epsilon"]:
            epsilon_callback = EpsilonCallback()
            callbacks_list += [epsilon_callback]

        # Test the trained model
        evaluationResults = self.model.evaluate(self.test_games, self.test_equilibria, batch_size=self.cfg["test_batch_size"], callbacks=callbacks_list)

        # Print max epsilon
        if self.cfg["compute_epsilon"]:
            print('max_epsilon:', tf.get_static_value(epsilon_callback.logs['max_epsilon']))

        # Save evaluation results
        pd.DataFrame([self.model.metrics_names, evaluationResults]).to_csv('./Reports/' + self.cfg["test_results_file"],
                                                                           index=False)

        # Commutativity test
        if self.cfg["commutativity_test_permutations"] > 0:
            average_mae = commutativity_test(tests_games=self.test_games,
                                             test_eq=self.test_equilibria,
                                             model=self.model,
                                             permutation_number=self.cfg["commutativity_test_permutations"],
                                             test_batch_size=self.cfg["test_batch_size"])
            print('Commutativity test finished. Average mean absolute error: ', average_mae, '\n')

        # Print examples
        self.printExamples(num_to_print)

    # ******
    def printExamples(self, num_to_print=None):
        """
        Function to print examples.
        """

        # If no num_to_print is provided, default to setting in cfg
        if not num_to_print:
            num_to_print = self.cfg["examples_to_print"]

        # Load test data if not already created
        if not (self.test_games.size and self.test_equilibria.size):
            self.test_games, self.test_equilibria = loadTestData(self.cfg["test_games_file"],
                                                                 self.cfg["test_equilibria_file"],
                                                                 self.cfg["max_equilibria"], self.cfg["num_players"],
                                                                 self.cfg["num_strategies"])

        # Print examples
        printExamples(numberOfExamples=num_to_print,
                      testSamples=self.test_games,
                      testEqs=self.test_equilibria,
                      nn_model=self.model,
                      examples_print_file=self.cfg["examples_print_file"],
                      pureStrategies_per_player=self.cfg["num_strategies"],
                      lossType=self.cfg["loss_type"],
                      payoffLoss_type=self.cfg["payoff_loss_type"],
                      num_players=self.cfg["num_players"],
                      enable_hydra=self.cfg["enable_hydra"],
                      cluster_examples=self.cfg["cluster_examples"],
                      payoffToEq_weight=self.cfg["payoff_to_equilibrium_weight"]
                      )

    # ******
    def load_datasets(self):
        """
        Function to read the game and equilibria data from dataset files.
        The input argument dataset_files should be an array (list, tuple, etc.) of arrays.
        Each inner array is a pair of (game_file, equilibrium_file).
        This function returns the games and equilibria in two separate numpy arrays.
        """

        # Set where to look for the dataset files
        dataset_directory = './Datasets/' + str(self.cfg["num_players"]) + 'P/' + str(self.cfg["num_strategies"][0])
        for strategy in self.cfg["num_strategies"][1:]:
            dataset_directory += 'x' + str(strategy)
        dataset_directory += '/'

        if not os.path.isdir(dataset_directory):
            print('\n\nError: The dataset directory does not exist.\n\n')
            exit()

        # List the file name lists of all different types of datasets
        datasetTypes = [self.cfg['general_dataset_files'], self.cfg['mixed_only_dataset_files'],
                        self.cfg['group_only_dataset_files'], self.cfg['mixed_group_only_dataset_files']]
        dataset_typeNames = ['general_dataset_files', 'mixed_only_dataset_files', 'group_only_dataset_files',
                             'mixed_group_only_dataset_files']

        # Read the dataset files
        firstTime = True
        for typeNumber, dataset_files in enumerate(datasetTypes):
            if len(dataset_files) > 0:
                firstDataset = dataset_files[0]
                sampleGames_currentType, sampleEquilibria_currentType = GetTrainingDataFromNPY(
                    dataset_directory + firstDataset[0] + '.npy', dataset_directory + firstDataset[1] + '.npy')

                for currentDataset in dataset_files[1:]:
                    current_sampleGames, current_sampleEquilibria = GetTrainingDataFromNPY(
                        dataset_directory + currentDataset[0] + '.npy', dataset_directory + currentDataset[1] + '.npy')
                    sampleGames_currentType = np.append(sampleGames_currentType, current_sampleGames, axis=0)
                    sampleEquilibria_currentType = np.append(sampleEquilibria_currentType, current_sampleEquilibria,
                                                             axis=0)

                if firstTime:
                    sampleGames = np.empty(
                        shape=(0, self.cfg["num_players"]) + tuple(self.cfg["num_strategies"]))
                    sampleEquilibria = np.empty(shape=(0, sampleEquilibria_currentType.shape[1],
                                                       self.cfg["num_players"], max(self.cfg["num_strategies"])))
                    firstTime = False

                # Extract the requested number of samples and combine the arrays of different types
                if self.cfg['dataset_types_quota'][typeNumber] > sampleGames_currentType.shape[0]:
                    raise Exception('\nThe requested quota for ' + dataset_typeNames[
                        typeNumber] + ' is more than the samples in the dataset files provided.\n')
                else:
                    sampleGames = np.append(sampleGames,
                                            sampleGames_currentType[: self.cfg['dataset_types_quota'][typeNumber]], axis=0)
                    sampleEquilibria = np.append(sampleEquilibria, sampleEquilibria_currentType[
                                                                   : self.cfg['dataset_types_quota'][typeNumber]], axis=0)
            elif self.cfg['dataset_types_quota'][typeNumber] > 0:
                raise Exception('\nThe requested quota for ' + dataset_typeNames[
                    typeNumber] + ' is more than zero, but no dataset files provided.\n')

        # Shuffle the dataset arrays
        sampleGames, sampleEquilibria = unisonShuffle(sampleGames, sampleEquilibria)

        # Limit the number of true equilibria for each sample game if they are more than max_equilibria
        if self.cfg["max_equilibria"] < sampleEquilibria.shape[1]:
            sampleEquilibria = sampleEquilibria[:, 0: self.cfg["max_equilibria"], :, :]
        elif self.cfg["max_equilibria"] > sampleEquilibria.shape[1]:
            raise Exception(
                '\nmax_equilibria is larger than the number of per sample true equilibria in the provided dataset.\n')

        # Normalize if set to do so
        if self.cfg["normalize_input_data"]:
            axes_except_zero = (ax for ax in range(1, len(sampleGames.shape)))
            scalar_dim_except_zero = (sampleGames.shape[0],) + (1,) * (len(sampleGames.shape) - 1)
            sampleGames = (sampleGames - np.reshape(np.min(sampleGames, axis=axes_except_zero), scalar_dim_except_zero)) / \
                          np.reshape(np.max(sampleGames, axis=axes_except_zero) -
                                       np.min(sampleGames, axis=axes_except_zero), scalar_dim_except_zero)

        # Extract the training and test data
        sample_no = sampleGames.shape[0]
        trainingSamples_no = int((1 - self.cfg["test_split"]) * sample_no)

        training_games = sampleGames[: trainingSamples_no]
        training_equilibria = sampleEquilibria[: trainingSamples_no]
        test_games = sampleGames[trainingSamples_no: sample_no]
        test_equilibria = sampleEquilibria[trainingSamples_no: sample_no]

        return training_games, training_equilibria, test_games, test_equilibria

    # ******
    def load_config(self, configFile, configSection="DEFAULT"):
        """
        Function to load the configuration stored in a file.
        """

        # Create a configuration dictionary
        self.cfg = dict()

        # Load config parser
        config_parser = configparser.ConfigParser()

        # Read the config file
        listOfFilesRead = config_parser.read('./Config/' + configFile)

        # Make sure at least a configuration file was read
        if len(listOfFilesRead) <= 0:
            raise Exception("Fatal Error: No configuration file " + configFile + " found in ./Config/.")

        # Load all the necessary configurations
        self.cfg["num_players"] = config_parser.getint(configSection, "num_players")
        self.cfg["num_strategies"] = ast.literal_eval(config_parser.get(configSection, "num_strategies"))
        self.cfg["max_equilibria"] = config_parser.getint(configSection, "max_equilibria")
        self.cfg["initial_model_weights"] = config_parser.get(configSection, "initial_model_weights")
        self.cfg["validation_split"] = config_parser.getfloat(configSection, "validation_split")
        self.cfg["test_split"] = config_parser.getfloat(configSection, "test_split")
        self.cfg["examples_to_print"] = config_parser.getint(configSection, "examples_to_print")
        self.cfg["epochs"] = config_parser.getint(configSection, "epochs")
        self.cfg["learning_rate_cycle_length"] = config_parser.getint(configSection, "learning_rate_cycle_length")
        self.cfg["initial_learning_rate"] = config_parser.getfloat(configSection, "initial_learning_rate")
        self.cfg["momentum"] = config_parser.getfloat(configSection, "momentum")
        self.cfg["nesterov"] = config_parser.getboolean(configSection, "nesterov")
        self.cfg["batch_size"] = config_parser.getint(configSection, "batch_size")
        self.cfg["normalize_input_data"] = config_parser.getboolean(configSection, "normalize_input_data")
        self.cfg['dataset_types_quota'] = ast.literal_eval(config_parser.get(configSection, "dataset_types_quota"))
        self.cfg['general_dataset_files'] = ast.literal_eval(config_parser.get(configSection, "general_dataset_files"))
        self.cfg['mixed_only_dataset_files'] = ast.literal_eval(
            config_parser.get(configSection, "mixed_only_dataset_files"))
        self.cfg['group_only_dataset_files'] = ast.literal_eval(
            config_parser.get(configSection, "group_only_dataset_files"))
        self.cfg['mixed_group_only_dataset_files'] = ast.literal_eval(
            config_parser.get(configSection, "mixed_group_only_dataset_files"))
        self.cfg["model_architecture_file"] = config_parser.get(configSection, "model_architecture_file")
        self.cfg["model_weights_file"] = config_parser.get(configSection, "model_weights_file")
        self.cfg["loss_type"] = config_parser.get(configSection, "loss_type")
        self.cfg["payoff_to_equilibrium_weight"] = config_parser.getfloat(configSection, "payoff_to_equilibrium_weight")
        self.cfg["enable_hydra"] = config_parser.getboolean(configSection, "enable_hydra")
        self.cfg["test_games_file"] = config_parser.get(configSection, "test_games_file")
        self.cfg["test_equilibria_file"] = config_parser.get(configSection, "test_equilibria_file")
        self.cfg["training_history_file"] = config_parser.get(configSection, "training_history_file")
        self.cfg["test_results_file"] = config_parser.get(configSection, "test_results_file")
        self.cfg["examples_print_file"] = config_parser.get(configSection, "examples_print_file")
        self.cfg["batch_normalization"] = config_parser.getboolean(configSection, "batch_normalization")
        self.cfg["payoff_loss_type"] = config_parser.get(configSection, "payoff_loss_type")
        self.cfg["hydra_physique"] = config_parser.get(configSection, "hydra_physique")
        self.cfg["model_best_weights_file"] = config_parser.get(configSection, "model_best_weights_file")
        self.cfg["cluster_examples"] = config_parser.getboolean(configSection, "cluster_examples")
        self.cfg["compute_epsilon"] = config_parser.getboolean(configSection, "compute_epsilon")
        self.cfg["test_batch_size"] = config_parser.getint(configSection, "test_batch_size")
        self.cfg["commutativity_test_permutations"] = config_parser.getint(configSection, "commutativity_test_permutations")
        self.cfg["rewrite_saved_test_data_if_model_weights_given"] = config_parser.get(configSection, "rewrite_saved_test_data_if_model_weights_given")

        # Check input configurations
        if not (0 < self.cfg["validation_split"] < 1):
            raise ValueError('Input configuration "validation_split" is not between 0 and 1.')

        if not (0 < self.cfg["test_split"] < 1):
            raise ValueError('Input configuration "test_split" is not between 0 and 1.')

        if self.cfg["epochs"] < self.cfg["learning_rate_cycle_length"]:
            raise ValueError('Number of epochs is less than the cycle length of learning rate.')

        # Adjust the number of eopchs if needed and set the number of cycles of learning rates
        if (self.cfg["epochs"] % self.cfg["learning_rate_cycle_length"]) != 0:
            self.cfg["epochs"] -= self.cfg["epochs"] % self.cfg["learning_rate_cycle_length"]
            self.cfg["num_cycles"] = int(self.cfg["epochs"] / self.cfg["learning_rate_cycle_length"])

            # Notify the new epoch number
            print(
                '\nNumber of epochs must be a multiple of learning_rate_cycle_length.\nNumber of epochs is automatically set to ' + str(
                    self.cfg["epochs"]) + '\n\n')
        else:
            self.cfg["num_cycles"] = int(self.cfg["epochs"] / self.cfg["learning_rate_cycle_length"])
