import pickle, os, csv, time, math, configparser, h5py, re, random, ast
import tensorflow as tf
import numpy as np
from NashNet_metrics import NashNet_Metrics
from NashNet_utils import build_model, unisonShuffle, GetTrainingDataFromNPY, create_sse_loss

tf.compat.v1.disable_eager_execution()
# Class - NashNet
# Expected behavior and necessary functions
#   __init__    - Initializes the class, given a config file or nothing?
#
#   .add_model - Adds an additional model to the ensemble
#
#   .train  - Trains one or several models
#       Input - config file containing datasets to train off
#   .evaluate   - Given set of data & labels, compute loss, metrics, etc
#   .bulk_predict   - Given set of data, predict nash equilibria
#   .predict    - Given a game, predict nash equilbria


# models = list()
# models[0] = {"trained":False, "model":tf.keras.Model}
# Some utility functions for working with config files
def str_to_list(in_str):
    out_list = ast.literal_eval(in_str)
    return out_list


def str_to_dict(in_str):
    out_dict = ast.literal_eval(in_str)
    return out_dict


class NashNet():
    def __init__(self, config=None, verbose=1):
        # Create configuration dict
        self.cfg = dict()

        # Set verbosity
        self.verbose = verbose

        # Create models list
        self.models = []

        # Load config or yell at user
        if config:
            # Load config
            self.load_config(config)
        else:
            # Yell at user
            raise ValueError("No configuration file provided.")

        for weight_path in self.cfg["model_weights"]:
            self.add_model(weight_path)

    def add_model(self, weights=None):
        # Check if loading weights or initializing from random
        # Create a model
        model = dict()
        model["graph"] = tf.Graph()
        with model["graph"].as_default():
            model["trained"] = False
            model["model"] = build_model(num_players=self.cfg["num_players"],
                                         pure_strategies_per_player=self.cfg["num_strategies"],
                                         max_equilibria=self.cfg["max_equilibria"],
                                         optimizer=tf.keras.optimizers.SGD(learning_rate=self.cfg["initial_learning_rate"],
                                                                           momentum=self.cfg["momentum"],
                                                                           nesterov=self.cfg["nesterov"]),
                                         metrics=self.cfg["training_metrics"],
                                         create_loss=create_sse_loss)

            # Load weights if provided, and set trianed flag to true
            if weights:
                model["trained"] = True
                model["model"].load_weights(weights)

        # Append model to self.models
        self.models.append(model)

    def train(self, data_files, force_retrain=False):
        # Load training data
        sample_games, sample_equilibria = self.load_data(data_files)

        # Iterate through untrained models and train them
        for i, model in enumerate(self.models):
            # Check if model has been trained yet, or if force_retrain flag is true
            if force_retrain or not model["trained"]:
                with model["graph"].as_default():
                    if self.verbose >= 1:
                        print("Training model", i + 1, "of", len(self.models))

                    # Train model i
                    model["model"].fit(sample_games, sample_equilibria,
                                       validation_split=self.cfg["val_split"],
                                       epochs=self.cfg["max_epochs"],
                                       batch_size=self.cfg["batch_size"],
                                       shuffle=self.cfg["shuffle_training_data"],
                                       callbacks=[NashNet_Metrics(initial_lr=self.cfg["initial_learning_rate"],
                                                                 num_cycles=self.cfg["num_cycles"],
                                                                 max_epochs=self.cfg["max_epochs"],
                                                                 save_dir=self.cfg["working_dir"] + "/saved_weights/model" + str(i) + "/",
                                                                 save_name="model" + str(i)
                                                                 )]
                                       )
                    # Save the model
                    model["model"].save_weights(
                        self.cfg["working_dir"] + "/saved_weights/model" + str(i) + "/model" + str(i) + "final_weights.h5")

                # Marks as trained
                model["trained"] = True
            else:
                if self.verbose >= 2:
                    print("Model", i + 1, "has already been trained.")

                # # Re-save it locally anyways
                # model["model"].save_weights(self.cfg["working_dir"] + "/weights_" + str(i) + ".h5")
        # That's all folks

    def evaluate(self, data_files, num_to_print=None):
        # If no num_to_print is provided, default to setting in cfg
        if not num_to_print:
            num_to_print = self.cfg["examples_to_print"]

        # Load evaluation data
        sample_games, sample_equilibria = self.load_data(data_files)

        # Store results for each model in a list
        loss_metrics = []
        examples = []

        # Evaluate for every model
        for i, model in enumerate(self.models):
            loss_metrics.append(model["model"].evaluate(sample_games,
                                                        sample_equilibria,
                                                        batch_size=self.cfg["batch_size"]))
            nash_predicted = model["model"].predict(sample_games[0:num_to_print]).astype('float32')
            nash_true = sample_equilibria[0:num_to_print].astype('float32')
            examples.append((nash_true, nash_predicted))

        for i in range(num_to_print):
            print("\n\n", sample_equilibria[i][0], "\n", sample_equilibria[i][1])
            # for e in examples:
            #     print("\t Predicted:\n",e[i])

    # Reads in data from a list of file paths
    #   Return as two numpy arrays - training_samples and training_eqs
    def load_data(self, data_files, train_mode=True):
        # Create arrays with predefined sizes self.cfg["num_strategies"]
        sampleGames = np.zeros((0, self.cfg["num_players"], self.cfg["num_strategies"], self.cfg["num_strategies"]))
        sampleEquilibria = np.zeros(
            (0, self.cfg["max_equilibria"], self.cfg["num_players"], self.cfg["num_strategies"]))

        # Set where to look for the dataset files
        directory = './Datasets/' + str(self.cfg["num_players"]) + 'P/' + str(self.cfg["num_strategies"]) + 'x' + str(
            self.cfg["num_strategies"]) + '/'
        if not os.path.isdir(directory):
            print('\n\nError: The dataset directory does not exist.\n\n')
            exit()

        # Split data_files into games_files and equilibria_files
        games_files = data_files[0]
        equilibria_files = data_files[1]

        # Read the dataset files in npy format
        for gamesDataset, equilibriaDataset in zip(games_files, equilibria_files):
            sampleGames_temp, sampleEquilibria_temp = GetTrainingDataFromNPY(directory + gamesDataset, directory + equilibriaDataset)
            sampleGames = np.append(sampleGames, sampleGames_temp, axis = 0)
            sampleEquilibria = np.append(sampleEquilibria, sampleEquilibria_temp, axis = 0)

        # Shuffle the dataset arrays
        if train_mode:
            sampleGames, sampleEquilibria = unisonShuffle(sampleGames, sampleEquilibria)

        # Normalize if set to do so
        if self.cfg["normalize_input_data"]:
            sampleGames = (sampleGames - np.reshape(np.min(sampleGames, axis=(1, 2, 3)), (sampleGames.shape[0], 1, 1, 1))) / \
                          np.reshape(np.max(sampleGames, axis=(1, 2, 3)) - np.min(sampleGames, axis=(1, 2, 3)),
                                     (sampleGames.shape[0], 1, 1, 1)
                                     )
        return sampleGames, sampleEquilibria

    def load_config(self, config_path):
        # Debug print
        if self.verbose >= 1:
            print("Loading configuration file.")

        # D is for default. Just a shortcut for accessing default section in config parser
        d: str = "DEFAULT"

        # Load into memory using config parser
        config_parser = configparser.ConfigParser()

        # Attempt to load config file
        # Success stores list of files successfully parsed. Throw error if empty
        success = config_parser.read(config_path)

        # Check and make sure that a configuration file was read
        if self.verbose >= 2:
            print("Success is :", success, " with length ", len(success))
        if len(success) <= 0:
            raise Exception("Fatal Error: No configuration file \'" + config_path + "\' found.")

        # Load all the information
        self.cfg["num_players"] = config_parser.getint(d, "num_players")
        self.cfg["num_strategies"] = config_parser.getint(d, "num_strategies")
        self.cfg["max_equilibria"] = config_parser.getint(d, "max_equilibria")
        self.cfg["working_dir"] = config_parser.get(d, "working_dir")
        self.cfg["model_weights"] = str_to_list(config_parser.get(d, "model_weights"))
        self.cfg["val_split"] = config_parser.getfloat(d, "val_split")
        self.cfg["examples_to_print"] = config_parser.getint(d, "examples_to_print")
        self.cfg["num_cycles"] = config_parser.getint(d, "num_cycles")
        self.cfg["cycle_length"] = config_parser.getint(d, "cycle_length")
        self.cfg["initial_learning_rate"] = config_parser.getfloat(d, "initial_learning_rate")
        self.cfg["momentum"] = config_parser.getfloat(d, "momentum")
        self.cfg["nesterov"] = config_parser.getboolean(d, "nesterov")
        self.cfg["batch_size"] = config_parser.getint(d, "batch_size")
        self.cfg["shuffle_training_data"] = config_parser.getboolean(d, "shuffle_training_data")
        self.cfg["normalize_input_data"] = config_parser.getboolean(d, "normalize_input_data")
        self.cfg["training_metrics"] = str_to_list(config_parser.get(d, "training_metrics"))

        # Derived things
        self.cfg["max_epochs"] = self.cfg["cycle_length"] * self.cfg["num_cycles"]
