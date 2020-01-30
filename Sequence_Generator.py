import numpy as np
import math
import os
from tensorflow.keras.utils import Sequence
from NashNet_utils import unisonShuffle

'''
 Specifications for NashSequence:
    NashSequence, inheriting from sequence, is designed to feed a dataset into a tf.keras sequential model
    
__init__(
    file_list - List of game and equilibria tuples in the following form:
        [ (G1, E1), (G2, E2) ... (G**, E**)]
        Each pair of games / equilibria is assumed to be correct.
        Should contain the relative path from *NashNet Dir* 
            ex: ./Datasets/2P/2x2/Formatted_Data/
    max_equilibria - int >0
        Maximum number of equilibria to keep. Based on number of heads the hydra model contains
        Also checks that the equlibria files provided are valid
    normalize_input_data - Bool, whether or not to normalize input data
    batch_size - int >0, The number of samples to feed in each batch

    The sequence requires that __init__, __len__, and __getitem__ are defined.
    Additionally, on_epoch_end may be defined, and is called at the end of every epoch
    
    __len__ returns the number of batches that will be provided
    __getitem__ returns the loaded data
    on_epoch_end re-shuffles the indices list, and resets counters   
'''

class NashSequence(Sequence):
    def __init__(self, directory, test_split, max_equilibria, normalize_input_data, batch_size=500000, file_len=5000):
        # Check to make sure things work
        assert batch_size % file_len == 0
        assert file_len == 5000

        # Get files, then sort them to ensure they match up
        files = os.listdir(directory)
        self.directory = directory
        self.equilibria_files = [self.directory + x for x in files if "Equilibria" in x]
        self.game_files = [self.directory + x for x in files if "Games" in x]
        self.equilibria_files.sort()
        self.game_files.sort()

        # Do batch size
        self.batch_size = batch_size
        self.file_len = file_len
        self.max_equilibria = max_equilibria
        self.normalize_input_data = normalize_input_data

        # Create numpy array to hold x data
        tmp_game = np.load(self.game_files[0])
        game_shape = tmp_game.shape[-3:]
        self.x = np.zeros((batch_size,) + game_shape)

        # Create numpy array to hold y data
        tmp_eq = np.load(self.equilibria_files[0])
        eq_shape = tmp_eq.shape[-3:]
        self.y = np.zeros((batch_size,) + eq_shape)

        # Create randomized array of indices to... randomize the order things are fed into the model
        self.training_files_no = int(len(self.game_files) * (1 - test_split))
        self.indices = np.arange(self.training_files_no)

        # Var to track how many games have been fed into the model
        self.samples_fed = 0

    def __len__(self):
        return int(math.floor(self.training_files_no * self.file_len / self.batch_size))

    def __getitem__(self, idx):
        for i in range(int(self.batch_size / self.file_len)):
            # Load the game and Equ
            g = np.load(self.directory + self.game_files[self.indices[self.samples_fed]])
            e = np.load(self.directory + self.equilibria_files[self.indices[self.samples_fed]])
            self.x[i*self.file_len:(i+1)*self.file_len] = g
            self.y[i * self.file_len:(i + 1) * self.file_len] = e
            self.samples_fed += 1

        # Process samples
        self.x, self.y = self.__process_data(self.x, self.y)

        return self.x, self.y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        self.samples_fed = 0

    def __save_test_data(self):
        pass

    def __process_data(self, sampleGames, sampleEquilibria):
        # Shuffle the dataset arrays
        sampleGames, sampleEquilibria = unisonShuffle(sampleGames, sampleEquilibria)

        # Limit the number of true equilibria for each sample game if they are more than max_equilibria
        if self.max_equilibria < sampleEquilibria.shape[1]:
            sampleEquilibria = sampleEquilibria[:, 0: self.max_equilibria, :, :]
        elif self.max_equilibria > sampleEquilibria.shape[1]:
            raise Exception(
                '\nmax_equilibria is larger than the number of per sample true equilibria in the provided dataset.\n')

        # Normalize if set to do so
        if self.normalize_input_data:
            axes_except_zero = tuple([ax for ax in range(1, len(sampleGames.shape))])
            scalar_dim_except_zero = (sampleGames.shape[0],) + (1,) * (len(sampleGames.shape) - 1)
            sampleGames = (sampleGames - np.reshape(np.min(sampleGames, axis=axes_except_zero), scalar_dim_except_zero)) / \
                          np.reshape(np.max(sampleGames, axis=axes_except_zero) - np.min(sampleGames, axis=axes_except_zero), scalar_dim_except_zero)

        return sampleGames, sampleEquilibria



