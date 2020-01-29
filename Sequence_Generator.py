import numpy as np
import math
import os
from tensorflow.keras.utils import Sequence

class NashSequence(Sequence):
    def __init__(self, directory, batch_size=500000, file_len=5000):
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

        # Create numpy array to hold x data
        tmp_game = np.load(self.game_files[0])
        game_shape = tmp_game.shape[-3:]
        self.x = np.zeros((batch_size,) + game_shape)

        # Create numpy array to hold y data
        tmp_eq = np.load(self.equilibria_files[0])
        eq_shape = tmp_eq.shape[-3:]
        self.y = np.zeros((batch_size,) + eq_shape)

        # Create randomized array of indices to... randomize the order things are fed into the model
        self.indices = np.arange(len(self.game_files))

        # Var to track how many games have been fed into the model
        self.samples_fed = 0

    def __len__(self):
        return math.ceil(len(self.game_files) * self.file_len / self.batch_size)

    def __getitem__(self, idx):
        for i in range(int(self.batch_size / self.file_len)):
            # Load the game and Equ
            g = np.load(self.directory + self.game_files[self.indices[self.samples_fed]])
            e = np.load(self.directory + self.equilibria_files[self.indices[self.samples_fed]])
            self.x[i*self.file_len:(i+1)*self.file_len] = g
            self.y[i * self.file_len:(i + 1) * self.file_len] = e
            self.samples_fed += 1
        return self.x, self.y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        self.samples_fed = 0


