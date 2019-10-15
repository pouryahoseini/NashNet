import tensorflow as tf
import math


class NashNet_Metrics(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, num_cycles, max_epochs, save_dir, save_name):
        # Learning Rate Scheduler Variables
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.num_cycles = num_cycles

        # Checkpoint and saving variables
        self.save_dir = save_dir
        self.save_name = save_name

        self.a = 0
    # Done

    def on_train_begin(self, logs=None):
        self.a = 1

    # Done

    def on_train_end(self, logs=None):
        self.a = 1

    # Done

    def on_epoch_begin(self, epoch, logs=None):
        # Calculate new learning rate
        new_lr = self.initial_lr / 2 * (math.cos(
            math.pi * ((epoch) % (self.max_epochs / self.num_cycles)) / (self.max_epochs / self.num_cycles)) + 1)

        # Update learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

    # Done

    # def on_epoch_end(self, epoch, logs=None):
    #     # If model is at minima, save the model
    #     if (epoch) % (self.max_epochs / self.num_cycles) == 0:
    #         # Get snapshot number
    #         snpashot_num = int(epoch / (self.max_epochs / self.num_cycles))
    #
    #         # Get path for saving the weights
    #         weights_path = self.save_dir + "/" + self.save_name + '_weights_snapshot' + str(snpashot_num) + '.h5'
    #
    #         # Save the weights
    #         self.model.save_weights(weights_path)
    #
        # Just save the fucking model anyways
        self.model.save_weights(self.save_dir + "/" + self.save_name + '_weights_epoch' + str(epoch) + '.h5')
    # Done

    # def on_batch_begin(self, batch, logs={}):
    #     self.a = 1
    # # Done
    #
    # def on_batch_end(self, batch, logs={}):
    #     self.a = 1
    # # Done
