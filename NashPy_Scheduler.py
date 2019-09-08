import numpy as np
import math
import keras
import tensorflow as tf
# # Learning rate scheduler - Sets learning rate for every epoch
# # Following "Cyclic Cosine Annealing" described in "Train 1 Get M" paper
# def CosineAnnealingLR(epoch, a0=1, max_epochs=500, num_cycles=10):
#     return a0/2 * (math.cos(math.pi*(epoch-1 % (max_epochs/num_cycles))/(max_epochs/num_cycles))+1)

# scheduler = tf.keras.callbacks.LearningRateScheduler(CosineAnnealingLR)

# # Returns a function which may be passed to tf.keras.callbacks.LearningRateScheduler
# def build_cosine_annealing_scheduler(initial_lr, num_epochs, num_cycles):
#     # The actual annealing function
#     def cosine_annealing(epoch):
#         # initial_lr/2






# Metrics Callback - Saves model during training
class Metrics(keras.callbacks.Callback):
    def __init__(self, initial_lr, max_epochs, num_cycles, save_dir, save_name):
        # Learning Rate Scheduler Variables
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.num_cycles = num_cycles

        # Checkpoint and saving variables
        self.save_dir = save_dir
        self.save_name = save_name

    def on_epoch_end(self, epoch, logs=None)
        # Calculate new learning rate
        new_lr = self.initial_lr/2 * (math.cos(math.pi*((epoch-1) % (self.max_epochs/self.num_cycles))/(self.max_epochs/self.num_cycles))+1)

        # Update learning rate
        self.model.optimizer.lr.set_value(new_lr)

        # If model is at minima, save the model
        if (epoch-1) % (self.max_epochs/self.num_cycles) == 0:
            # Get snapshot number
            snpashot_num = int(epoch/(self.max_epochs/self.num_cycles))


            # Saving model architecture is not necessary! Only save the model weights when
            # taking a snapshot.
            # MAKE SURE TOS AVE ARCHITECTURE SOMEWHERE!!!!!!!!!!!!!!!!!!

            # # Get path of model
            # save_path = self.save_dir + "/" + self.save_name + '_' + str(snpashot_num) + '.json'

            # # Write the model to file
            # with open(save_path) as json_file::
            #     json_file.write(self.model.to_json())
            
            # Get path for saving the weights
            weights_path = self.save_dir + "/" + self.save_name + '_weights_' + str(snpashot_num) + '.h5'

            # Save the weights
            self.model.save_weights(weights_path)




