import numpy as np
import math
import os
from tensorflow.keras.utils import Sequence
from NashNet_utils import unisonShuffle


class NashSequence(Sequence):
    def __init__(self, files_list, files_location, max_equilibria, normalize_input_data, batch_size=64):
        """
        Constructor for the dataset generator.
        :param files_list: List of game and equilibria tuples in the following form:
        [ (G1, E1), (G2, E2) ... (G**, E**)]
        Each pair of games / equilibria is assumed to be correct.
        Should contain the relative path from *NashNet Dir*
            ex: ./Datasets/2P/2x2/Formatted_Data/
        :param files_location: The location to *NashNet Dir*
        :param max_equilibria: int >0
        Maximum number of equilibria to keep. Based on number of heads the hydra model contains
        Also checks that the equilibria files provided are valid
        :param normalize_input_data: Bool, whether or not to normalize input data
        :param batch_size: int >0, The number of samples to feed in each batch
        """

        # Get files, then sort them to ensure they match up
        self.game_files = [x[0] for x in files_list]
        self.equilibria_files = [x[1] for x in files_list]

        # Do batch size
        self.batch_size = batch_size
        self.max_equilibria = max_equilibria
        self.normalize_input_data = normalize_input_data
        self.files_location = files_location
        self.last_file_index = -1

        # Create numpy array to hold x data
        tmp_game = np.load(os.path.join(files_location, self.game_files[0]))
        game_shape = tmp_game.shape[1:]
        self.x = np.zeros((batch_size,) + game_shape)

        # Create numpy array to hold y data
        tmp_eq = np.load(os.path.join(files_location, self.equilibria_files[0]))
        eq_shape = tmp_eq.shape[1:]
        self.y = np.zeros((batch_size,) + eq_shape)

        # Check to make sure that each file is the same size
        #   this is necessary in order to make this properly thread-safe
        #   so that we can use multiple workers to prefetch batches faster
        self.file_len = tmp_game.shape[0]

        # NOTE! Currently, a batch cannot span over more than two files.
        # This means batch size cannot be greater than the file length
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
            self.num_samples = self.num_samples + g.shape[0]

        # Create array of indices to randomize the order things are fed into the model
        self.num_training_files = len(self.game_files)
        self.indices = np.arange(self.num_training_files)
        np.random.shuffle(self.indices)

    # Gets number of batches
    def __len__(self):
        """
        :return: The number of batches that will be provided
        """

        return int(math.ceil(self.num_samples / self.batch_size))

    # Gets the batch, specified by batch_num
    def __getitem__(self, batch_num):
        """
        Function to load and return a batch of samples
        :param batch_num: The number of samples in the batch
        :return: The loaded batch of data
        """

        # 2 Cases - Data is all in one file, or data is split over two files
        #   1st file is found by using the formula f1_idx = int(batch_num*self.batch_size/self.file_len)
        f1_idx = int(batch_num * self.batch_size / self.file_len)
        lower = (batch_num * self.batch_size) % self.file_len
        upper = ((batch_num + 1) * self.batch_size) % self.file_len

        # Load samples from f1 - This will always be done.
        if self.last_file_index != f1_idx:
            self.g = np.load(os.path.join(self.files_location, self.game_files[self.indices[f1_idx]]))
            self.e = np.load(os.path.join(self.files_location, self.equilibria_files[self.indices[f1_idx]]))
            self.last_file_index = f1_idx

        # If lower > upper, then two files needed
        if lower > upper:
            remainder = self.batch_size - upper
            f2_idx = f1_idx + 1

            # If f2_idx >= self.num_training_files, then the end has been reached, and this is a special case
            # In this case, copy to the variables as much as possible and let the rest unchanged
            if f2_idx >= self.num_training_files:
                self.x[0: self.file_len - lower] = np.copy(self.g[lower: self.file_len])
                self.y[0: self.file_len - lower] = np.copy(self.e[lower: self.file_len])

            # If not, then things are normal
            else:
                # Assign stuff from file 1
                self.x[0:remainder] = self.g[lower: self.file_len]
                self.y[0:remainder] = self.e[lower: self.file_len]

                # Load f2
                self.g = np.load(os.path.join(self.files_location, self.game_files[self.indices[f2_idx]]))
                self.e = np.load(os.path.join(self.files_location, self.equilibria_files[self.indices[f2_idx]]))
                self.last_file_index = f2_idx

                # Assign the rest of the values to x and y
                self.x[remainder:self.batch_size] = self.g[0: upper]
                self.y[remainder:self.batch_size] = self.e[0: upper]

        # Only one file needed
        else:
            self.x = self.g[lower: upper]
            self.y = self.e[lower: upper]

        # Process samples and return
        return self.__process_data(self.x, self.y)

    def on_epoch_end(self):
        """
        re-shuffles the indices list of data files
        """

        np.random.shuffle(self.indices)

    def __process_data(self, sampleGames, sampleEquilibria):
        """
        Function to process the games and equilibria before passing it to the optimizer
        """

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
