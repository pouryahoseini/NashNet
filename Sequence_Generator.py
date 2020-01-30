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
    def __init__(self, files_list, files_location, max_equilibria, normalize_input_data, batch_size=64):
        # Get files, then sort them to ensure they match up
        self.game_files = [x[0] for x in files_list]
        self.equilibria_files = [x[1] for x in files_list]

        # Do batch size
        self.batch_size = batch_size
        self.max_equilibria = max_equilibria
        self.normalize_input_data = normalize_input_data
        self.files_location = files_location

        # Create numpy array to hold x data
        tmp_game = np.load(os.path.join(files_location, self.game_files[0]))
        game_shape = tmp_game.shape[1:]
        self.x = np.zeros((batch_size,) + game_shape)

        # Create numpy array to hold y data
        tmp_eq = np.load(os.path.join(files_location, self.equilibria_files[0]))
        eq_shape = tmp_eq.shape[1:]
        self.y = np.zeros((batch_size,) + eq_shape)

        # Check to make sure that each file is the SAME DAMN SIZE
        #   this is necessary in order to make this shit properly threadsafe
        #   so that we can use multiple workers to prefetch batches faster
        #   Makes the tiny-ass batch size such a bit less
        self.file_len = tmp_game.shape[0]

        # NOTE! Currently, a batch cannot span over more than two files.
        # This means batch size CANNOT be greater than the file len
        if self.batch_size > self.file_len:
            raise ValueError("Err: Batch size cannot be greater than number of samples in a single file!")

        self.num_samples = 0
        for gf in self.game_files:
            g = np.load(os.path.join(files_location, gf))
            # Check shape
            if g.shape[0] != self.file_len:
                errmsg = "All data files must be the same size!\n\
                           \tFile " + gf + " has shape " + str(g.shape) + \
                          "\twhile file " + self.game_files[0] + " has shape " + str(game_shape) + "!"
                raise ValueError(errmsg)

            # Accumulate
            self.num_samples += self.num_samples + g.shape[0]

        # Create array of indices to randomize the order things are fed into the model
        self.num_training_files = len(self.game_files)
        self.indices = np.arange(self.num_training_files)
        np.random.shuffle(self.indices)

    # Gets number of batches
    def __len__(self):
        return int(math.ceil(self.num_samples / self.batch_size))

    # Gets the batch, specified by batch_num
    def __getitem__(self, batch_num):
        # 2 Cases - Data is all in one file, or data is split over two files
        #   1st file is found by using the formula f1_idx = int(batch_num*self.batch_size/self.file_len)
        f1_idx = int(batch_num*self.batch_size/self.file_len)
        lower=batch_num * self.batch_size % self.file_len
        upper=(batch_num+1) * self.batch_size % self.file_len

        # Load samples from f1 - This will always be done.
        g = np.load(os.path.join(self.files_location, self.game_files[f1_idx]))
        e = np.load(os.path.join(self.files_location, self.equilibria_files[f1_idx]))

        # If lower > upper, then two files needed
        if lower > upper:
            remainder = self.file_len - lower
            f2_idx = f1_idx+1

            # If f2_idx >= self.num_training_files, then the end has been reached, and this is a special case
            if f2_idx >= self.num_training_files:
                self.x = np.copy(g[lower:])
                self.y = np.copy(e[lower:])

            # If not, then things are normal
            else:
                # Assign stuff from file 1
                self.x[0:remainder] = g[lower:]
                self.y[0:remainder] = e[lower:]

                # Load f2
                g = np.load(os.path.join(self.files_location, self.game_files[f1_idx]))
                e = np.load(os.path.join(self.files_location, self.equilibria_files[f1_idx]))

                # Assign the rest of the values to x and y
                self.x[remainder:self.batch_size] = g[0:remainder]
                self.y[remainder:self.batch_size] = e[0:remainder]

        # Only one file needed
        else:
            self.x[0:self.batch_size] = g[lower:upper]
            self.y[0:self.batch_size] = e[lower:upper]

        # Process samples
        self.x, self.y = self.__process_data(self.x, self.y)

        return self.x, self.y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

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



