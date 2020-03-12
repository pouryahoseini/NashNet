import configparser, ast
import random
import tensorflow.keras as keras
from Sequence_Generator import *
from NashNet_utils import *
import logging
from Network_Topology import build_hydra_model, build_monohead_model
from Test_Metrics import epsilon_test, commutativity_test


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
        self.test_files = None
        self.trained_model_loaded = False

        # Set Tensorflow's verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # WARN
        logging.getLogger('tensorflow').setLevel(logging.WARN)

        # Load config
        self.load_config(configFile, configSection=configSection)

        # Find the address to the dataset
        self.dataset_location = self.dataset_address()

        # Build the neural network model
        self.build_model()

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

        # Decide on the training, validation, and test data lists of files
        training_files, validation_files, self.test_files = self.list_files(validation_split=self.cfg["validation_split"],
                                                                            test_split=self.cfg["test_split"])

        # Overwrite the list of test files if an initial training is available and there is a test files list file
        if self.weights_initialized:
            try:
                self.test_files = loadDataSplit(saved_files_list=self.cfg["test_files_list"],
                                                data_split_folder=self.cfg["data_split_folder"],
                                                num_players=self.cfg["num_players"],
                                                num_strategies=self.cfg["num_strategies"])
                validation_files = loadDataSplit(saved_files_list=self.cfg["validation_files_list"],
                                                 data_split_folder=self.cfg["data_split_folder"],
                                                 num_players=self.cfg["num_players"],
                                                 num_strategies=self.cfg["num_strategies"])
                training_files = loadDataSplit(saved_files_list=self.cfg["training_files_list"],
                                               data_split_folder=self.cfg["data_split_folder"],
                                               num_players=self.cfg["num_players"],
                                               num_strategies=self.cfg["num_strategies"])
            except FileNotFoundError:
                print("Data split information were not available, deciding on the test data now.")

        # Save the list of test files
        if self.cfg['save_split_data'] and (self.cfg["rewrite_saved_test_data_if_model_weights_given"] or not self.weights_initialized):
            saveDataSplit(files_list=self.test_files,
                          save_list_file=self.cfg["test_files_list"],
                          data_split_folder=self.cfg["data_split_folder"],
                          num_players=self.cfg["num_players"],
                          num_strategies=self.cfg["num_strategies"])
            saveDataSplit(files_list=validation_files,
                          save_list_file=self.cfg["validation_files_list"],
                          data_split_folder=self.cfg["data_split_folder"],
                          num_players=self.cfg["num_players"],
                          num_strategies=self.cfg["num_strategies"])
            saveDataSplit(files_list=training_files,
                          save_list_file=self.cfg["training_files_list"],
                          data_split_folder=self.cfg["data_split_folder"],
                          num_players=self.cfg["num_players"],
                          num_strategies=self.cfg["num_strategies"])

        # Print the summary of the model
        print(self.model.summary())

        # Create the list of callbacks
        callbacks_list = [TrainingCallback(initial_lr=self.cfg["initial_learning_rate"],
                                           num_cycles=self.cfg["num_cycles"],
                                           max_epochs=self.cfg["epochs"],
                                           save_dir='./Model/Interim/',
                                           save_name="interimWeight",
                                           save_interim_weights=self.cfg["save_interim_weights"]),
                          keras.callbacks.ModelCheckpoint(filepath='./Model/' + self.cfg[
                              "model_best_weights_file"] + '.h5',
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          mode='min',
                                                          save_freq='epoch')]

        # Train the model
        seq = NashSequence(files_list=training_files,
                           files_location=self.dataset_location,
                           max_equilibria=self.cfg["max_equilibria"],
                           normalize_input_data=self.cfg["normalize_input_data"],
                           batch_size=self.cfg["batch_size"])
        valid_seq = NashSequence(files_list=validation_files,
                                 files_location=self.dataset_location,
                                 max_equilibria=self.cfg["max_equilibria"],
                                 normalize_input_data=self.cfg["normalize_input_data"],
                                 batch_size=self.cfg["batch_size"])
        trainingHistory = self.model.fit(seq,
                                         validation_data=valid_seq,
                                         epochs=self.cfg["epochs"],
                                         shuffle='batch',
                                         callbacks=callbacks_list,
                                         max_queue_size=self.cfg['generator_max_queue_size'],
                                         use_multiprocessing=self.cfg['generator_multiprocessing']
                                         )

        # Save the model
        saveModel(model=self.model,
                  model_architecture_file=self.cfg["model_architecture_file"],
                  model_weights_file=self.cfg["model_weights_file"])

        # Write the loss and metric values during the training and test time
        save_training_history(trainingHistory, self.cfg["training_history_file"])

        # Load best-found weights if it is enabled
        if self.cfg["use_best_weights"]:
            self.model.load_weights(os.path.join('./Model/' + self.cfg["model_best_weights_file"] + '.h5'))

        # Indicate trained model is ready
        self.trained_model_loaded = True

    # ******
    def evaluate(self, num_to_print=None):
        """
        Function to evaluate NashNet.
        """

        # Load model weights if the model is not loaded yet
        if not self.trained_model_loaded:
            if self.cfg["use_best_weights"]:  # Use best weights in the training
                self.model.load_weights(os.path.join('./Model/' + self.cfg["model_best_weights_file"] + '.h5'))
            else:  # Use last weights in the training
                self.model.load_weights(os.path.join('./Model/' + self.cfg["model_weights_file"] + '.h5'))
            self.trained_model_loaded = True

        # If no num_to_print is provided, default to setting in cfg
        if not num_to_print:
            num_to_print = self.cfg["examples_to_print"]

        # Load list of test data files if not already created
        if self.test_files is None:
            self.test_files = loadDataSplit(saved_files_list=self.cfg["test_files_list"],
                                            data_split_folder=self.cfg["data_split_folder"],
                                            num_players=self.cfg["num_players"],
                                            num_strategies=self.cfg["num_strategies"])

        # Create the test data generator
        test_seq = NashSequence(files_list=self.test_files,
                                files_location=self.dataset_location,
                                max_equilibria=self.cfg["max_equilibria"],
                                normalize_input_data=self.cfg["normalize_input_data"],
                                batch_size=self.cfg["test_batch_size"])

        # Create the list of callbacks
        callbacks_list = []

        # Test the trained model
        evaluationResults = self.model.evaluate(test_seq,
                                                callbacks=callbacks_list,
                                                max_queue_size=self.cfg['generator_max_queue_size'],
                                                use_multiprocessing=self.cfg['generator_multiprocessing']
                                                )

        # Save the list of model metrics
        model_metrics = self.model.metrics_names

        # Print max epsilon and remove the per-epoch epsilon
        if self.cfg["compute_epsilon"]:
            max_epsilon, max_delta, mean_epsilon, mean_delta = epsilon_test(
                test_data_generator=test_seq,
                model=self.model,
                pureStrategies_perPlayer=self.cfg["num_strategies"],
                num_players=self.cfg["num_players"],
                lossType=self.cfg['loss_type'],
                payoffLoss_type=self.cfg['payoff_loss_type'],
                hydra_enabled=self.cfg["enable_hydra"])

            # Add max epsilon and max delta to the results
            print('max_epsilon: ', max_epsilon, '\nmax_delta: ', max_delta, '\nmean_epsilon: ', mean_epsilon, '\nmean_delta: ', mean_delta)
            model_metrics += ['max_epsilon', 'mean_epsilon', 'max_delta', 'mean_delta']
            evaluationResults += [max_epsilon, mean_epsilon, max_delta, mean_delta]

        # Commutativity test
        if self.cfg["commutativity_test_permutations"] > 0:
            average_mae = commutativity_test(test_data_generator=test_seq,
                                             model=self.model,
                                             permutation_number=self.cfg["commutativity_test_permutations"]
                                             )

            model_metrics += ['Commutativity average MAE']
            evaluationResults += [average_mae]
            print('Commutativity test finished. Average mean absolute error: ', average_mae, '\n')

        # Save evaluation results
        pd.DataFrame([model_metrics, evaluationResults]).to_csv('./Reports/' + self.cfg["test_results_file"], index=False)

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

            # Load list of test data files if not already created
            if not self.test_files:
                self.test_files = loadDataSplit(saved_files_list=self.cfg["test_files_list"],
                                                data_split_folder=self.cfg["data_split_folder"],
                                                num_players=self.cfg["num_players"],
                                                num_strategies=self.cfg["num_strategies"])

        # Create the test data generator
        test_seq = NashSequence(files_list=self.test_files,
                                files_location=self.dataset_location,
                                max_equilibria=self.cfg["max_equilibria"],
                                normalize_input_data=self.cfg["normalize_input_data"],
                                batch_size=self.cfg["test_batch_size"])

        # Print examples
        printExamples(numberOfExamples=num_to_print,
                      test_data_generator=test_seq,
                      nn_model=self.model,
                      examples_print_file=self.cfg["examples_print_file"],
                      pureStrategies_per_player=self.cfg["num_strategies"],
                      lossType=self.cfg["loss_type"],
                      payoffLoss_type=self.cfg["payoff_loss_type"],
                      num_players=self.cfg["num_players"],
                      enable_hydra=self.cfg["enable_hydra"],
                      cluster_examples=self.cfg["cluster_examples"],
                      print_to_terminal=self.cfg["print_to_terminal"],
                      payoffToEq_weight=self.cfg["payoff_to_equilibrium_weight"]
                      )

    # ******
    def dataset_address(self):
        """
        Function to get the address to the desired dataset
        """

        dataset_directory = './Datasets/' + str(self.cfg["num_players"]) + 'P/' + str(self.cfg["num_strategies"][0])
        for strategy in self.cfg["num_strategies"][1:]:
            dataset_directory += 'x' + str(strategy)
        dataset_directory += '/'

        if self.cfg['split_files_folder_name'] != '':
            dataset_directory += (self.cfg['split_files_folder_name'] + '/')

        return dataset_directory

    # ******
    def generate_test_data_array(self, test_files):
        """
        Function to generate a numpy array of test data from the list of test files
        """

        # Go to the proper address
        address = self.dataset_address()

        # Load and concatenate the files
        test_games = np.load(address + test_files[0][0])
        test_equilibria = np.load(address + test_files[0][1])

        for (game_file, eq_file) in test_files[1:]:
            games_temp = np.load(address + game_file)
            eq_temp = np.load(address + eq_file)
            test_games = np.append(test_games, games_temp, axis=0)
            test_equilibria = np.append(test_equilibria, eq_temp, axis=0)

        # Limit the number of true equilibria for each sample game if they are more than max_equilibria
        if self.cfg["max_equilibria"] < test_equilibria.shape[1]:
            test_equilibria = test_equilibria[:, 0: self.cfg["max_equilibria"], :, :]
        elif self.cfg["max_equilibria"] > test_equilibria.shape[1]:
            raise Exception(
                '\nmax_equilibria is larger than the number of per sample true equilibria in the provided dataset.\n')

        # Normalize if set to do so
        if self.cfg["normalize_input_data"]:
            axes_except_zero = tuple([ax for ax in range(1, len(test_games.shape))])
            scalar_dim_except_zero = (test_games.shape[0],) + (1,) * (len(test_games.shape) - 1)
            test_games = (test_games - np.reshape(np.min(test_games, axis=axes_except_zero), scalar_dim_except_zero)) / \
                          np.reshape(np.max(test_games, axis=axes_except_zero) - np.min(test_games, axis=axes_except_zero), scalar_dim_except_zero)

        return test_games, test_equilibria

    # ******
    def list_files(self, validation_split, test_split):
        """
        Function to list the file names for training, testing, and evaluating with the format
        [(G1, E1), (G2, E2) ... (G**, E**)]
        """

        # Find the address to the files
        address = self.dataset_address()

        # List the files in the directory
        list_of_files = os.listdir(address)

        # Extract the list of training files
        equilibrium_files = [file for file in list_of_files if "Equilibria" in file]
        game_files = [file for file in list_of_files if "Games" in file]
        equilibrium_files.sort()
        game_files.sort()

        # Check if the entries in the list of equilibria and games are corresponding to each other
        assert len(game_files) == len(equilibrium_files), 'The number of game and equilibrium files are not equal in ' + address
        for file_no, file in enumerate(game_files):
            assert file.lstrip('Games_') == equilibrium_files[file_no].lstrip('Equilibria_'), \
                'The equilibrium and game files in the directory ' + address + ' are not matching.\nMismatched file: ' + file

        # Shuffle the list of files
        zipped_list = list(zip(game_files, equilibrium_files))
        random.shuffle(zipped_list)
        game_files, equilibrium_files = zip(*zipped_list)

        # Set the starting index of validation and test files
        test_index = int(math.floor(len(game_files) * (1 - test_split)))
        validation_index = int(math.floor(test_index * (1 - validation_split)))

        # Create the final list of test, training, and validation files
        training_files = [(game_file, equilibrium_file) for (game_file, equilibrium_file) in zip(game_files[: validation_index], equilibrium_files[: validation_index])]
        validation_files = [(game_file, equilibrium_file) for (game_file, equilibrium_file) in zip(game_files[validation_index: test_index], equilibrium_files[validation_index: test_index])]
        test_files = [(game_file, equilibrium_file) for (game_file, equilibrium_file) in zip(game_files[test_index:], equilibrium_files[test_index:])]

        return training_files, validation_files, test_files

    # ******
    def build_model(self):
        """
        Function to build and compile the neural network model
        """

        # Build the neural network model
        if not self.cfg["enable_hydra"]:
            self.model = build_monohead_model(num_players=self.cfg["num_players"],
                                              pure_strategies_per_player=self.cfg["num_strategies"],
                                              max_equilibria=self.cfg["max_equilibria"],
                                              optimizer=tf.keras.optimizers.SGD(
                                                  learning_rate=self.cfg["initial_learning_rate"],
                                                  momentum=self.cfg["momentum"],
                                                  nesterov=self.cfg["nesterov"]),
                                              lossType=self.cfg["loss_type"],
                                              payoffLoss_type=self.cfg["payoff_loss_type"],
                                              monohead_common_layer_sizes=self.cfg["monohead_common_layer_sizes"],
                                              monohead_layer_sizes_per_player=self.cfg[
                                                  "monohead_layer_sizes_per_player"],
                                              enable_batchNormalization=self.cfg["batch_normalization"],
                                              payoffToEq_weight=self.cfg["payoff_to_equilibrium_weight"]
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
                                           sawfish_common_layer_sizes=self.cfg["sawfish_common_layer_sizes"],
                                           bull_necked_common_layer_sizes=self.cfg["bull_necked_common_layer_sizes"],
                                           sawfish_head_layer_sizes=self.cfg["sawfish_head_layer_sizes"],
                                           bull_necked_head_layer_sizes=self.cfg["bull_necked_head_layer_sizes"],
                                           hydra_layer_sizes_per_player=self.cfg["hydra_layer_sizes_per_player"],
                                           enable_batchNormalization=self.cfg["batch_normalization"],
                                           hydra_shape=self.cfg["hydra_physique"],
                                           payoffToEq_weight=self.cfg["payoff_to_equilibrium_weight"]
                                           )

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
        listOfFilesRead = config_parser.read(configFile)

        # Make sure at least a configuration file was read
        if len(listOfFilesRead) <= 0:
            raise Exception("Fatal Error: No configuration file " + configFile + " found in ./Configs/.")

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
        self.cfg["model_architecture_file"] = config_parser.get(configSection, "model_architecture_file")
        self.cfg["model_weights_file"] = config_parser.get(configSection, "model_weights_file")
        self.cfg["loss_type"] = config_parser.get(configSection, "loss_type")
        self.cfg["payoff_to_equilibrium_weight"] = config_parser.getfloat(configSection, "payoff_to_equilibrium_weight")
        self.cfg["enable_hydra"] = config_parser.getboolean(configSection, "enable_hydra")
        self.cfg["test_files_list"] = config_parser.get(configSection, "test_files_list")
        self.cfg["training_files_list"] = config_parser.get(configSection, "training_files_list")
        self.cfg["validation_files_list"] = config_parser.get(configSection, "validation_files_list")
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
        self.cfg["rewrite_saved_test_data_if_model_weights_given"] = config_parser.getboolean(configSection, "rewrite_saved_test_data_if_model_weights_given")
        self.cfg['sawfish_common_layer_sizes'] = ast.literal_eval(config_parser.get(configSection, "sawfish_common_layer_sizes"))
        self.cfg['bull_necked_common_layer_sizes'] = ast.literal_eval(config_parser.get(configSection, "bull_necked_common_layer_sizes"))
        self.cfg['monohead_common_layer_sizes'] = ast.literal_eval(config_parser.get(configSection, "monohead_common_layer_sizes"))
        self.cfg['sawfish_head_layer_sizes'] = ast.literal_eval(config_parser.get(configSection, "sawfish_head_layer_sizes"))
        self.cfg['bull_necked_head_layer_sizes'] = ast.literal_eval(config_parser.get(configSection, "bull_necked_head_layer_sizes"))
        self.cfg['hydra_layer_sizes_per_player'] = ast.literal_eval(config_parser.get(configSection, "hydra_layer_sizes_per_player"))
        self.cfg['monohead_layer_sizes_per_player'] = ast.literal_eval(config_parser.get(configSection, "monohead_layer_sizes_per_player"))
        self.cfg['split_files_folder_name'] = config_parser.get(configSection, "split_files_folder_name")
        self.cfg['save_split_data'] = config_parser.getboolean(configSection, "save_split_data")
        self.cfg['generator_max_queue_size'] = config_parser.getint(configSection, "generator_max_queue_size")
        self.cfg['generator_multiprocessing'] = config_parser.getboolean(configSection, "generator_multiprocessing")
        self.cfg["print_to_terminal"] = config_parser.getboolean(configSection, "print_to_terminal")
        self.cfg["save_interim_weights"] = config_parser.getboolean(configSection, "save_interim_weights")
        self.cfg["use_best_weights"] = config_parser.getboolean(configSection, "use_best_weights")
        self.cfg['data_split_folder'] = config_parser.get(configSection, "data_split_folder")

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
