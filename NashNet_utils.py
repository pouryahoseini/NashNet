import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import math
import pandas as pd
from sklearn import cluster
import os

# # ********************************
# def GetTrainingDataFromNPY(data_file, labels_file):
#     """
#     Function to load dataset from two .npy files
#     """
#
#     data = np.load(data_file)
#     labels = np.load(labels_file)
#
#     return data, labels


# ********************************
def unisonShuffle(a, b):
    """
    Function to shuffle two numpy arrays in unison
    """

    assert a.shape[0] == b.shape[0]

    p = np.random.permutation(a.shape[0])

    return a[p], b[p]


# ********************************
def MSE(nashEq_true, nashEq_pred):
    """
    Function to compute the correct mean squared error (mse) as a metric during model training and testing
    """

    # Compute error
    error = nashEq_true - nashEq_pred

    # Compute the weight for the final result to recompense the replacement of nans with zeros and its effect
    # on the averaging
    nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(nashEq_true[0][0]), tf.int32))
    eq_n_elements = tf.size(nashEq_true[0][0])
    compensation_factor = tf.cast(eq_n_elements / (eq_n_elements - nan_count), tf.float32)

    # Replace nan values with 0
    error = tf.where(tf.math.is_nan(error), tf.zeros_like(error), error)

    return K.mean(K.min(K.mean(K.square(error), axis=[2, 3]), axis=1)) * compensation_factor


# ********************************
def hydra_MSE(nashEq_true, nashEq_pred):
    """
    Function to compute the mean squared error (mse) as a metric during the hydra model training and testing
    """

    # Run the matching with the true and predicted equilibria as proposer and proposed (in terms of the Deferred Acceptance Algorithm terms) each time
    eq_MSE_trueProposed = hydra_oneSided_MSE(nashEq_pred, nashEq_true)
    eq_MSE_trueProposer = hydra_oneSided_MSE(nashEq_true, nashEq_pred)

    return K.mean((eq_MSE_trueProposer + eq_MSE_trueProposed) / 2.0)


# ********************************
def hydra_oneSided_MSE(nashEq_proposer, nashEq_proposed):
    """
    Function to compute MSE of equilibria with a proposer and proposed defined (in Deferred Acceptance Algorithm terms).
    """

    # Create a row-wise meshgrid of proposed equilibria for each sample in the batch by adding a new dimension and replicate the array along that
    proposed_grid = tf.tile(tf.expand_dims(nashEq_proposed, axis=2), [1, 1, tf.shape(nashEq_proposed)[1], 1, 1])

    # Create a column-wise meshgrid of proposer equilibria for each sample in the batch by adding a new dimension and replicate the array along that
    proposer_grid = tf.tile(tf.expand_dims(nashEq_proposer, axis=1), [1, tf.shape(nashEq_proposer)[1], 1, 1, 1])

    # Compute the weight for the final result to recompense the replacement of nans with zeros and its effect
    # on the averaging
    nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(nashEq_proposer[0][0] + nashEq_proposed[0][0]), tf.int32))
    eq_n_elements = tf.size(nashEq_proposer[0][0])
    compensation_factor = tf.cast(eq_n_elements / (eq_n_elements - nan_count), tf.float32)

    # Compute error grid
    error_grid = proposed_grid - proposer_grid

    # Replace nan values with 0
    error_grid = tf.where(tf.math.is_nan(error_grid), tf.zeros_like(error_grid), error_grid)

    return K.max(K.min(K.mean(K.square(error_grid), axis=[3, 4]), axis=2), axis=1) * compensation_factor


# ********************************
@tf.function
def computePayoff_np(game, equilibrium, pureStrategies_perPlayer, playerNumber):
    """
    Function to compute the payoff each player gets with the input equilibrium and the input game
    (games with more than 2 players).
    """

    # Extract mix strategies of each player
    mixStrategies_perPlayer = [tf.gather(equilibrium, pl, axis=1)[:, : pureStrategies_perPlayer[pl]] for pl in range(playerNumber)]
    playerProbShape_grid = tuple(np.ones(playerNumber) + np.identity(playerNumber) * (np.array(pureStrategies_perPlayer) - 1))
    playerProbShape = [tuple((tf.shape(mixStrategies_perPlayer[0])[0],) + tuple(playerProbShape_grid[pl])) for pl in
                       range(playerNumber)]
    mixStrategies_perPlayer = [K.reshape(mixStrategies_perPlayer[pl], playerProbShape[pl]) for pl in
                               range(playerNumber)]

    # Multiply them together to get the probability matrix
    probability_mat = mixStrategies_perPlayer[0]
    for pl in range(1, playerNumber):
        probability_mat *= mixStrategies_perPlayer[pl]

    # Adding a new dimension
    probability_mat = K.expand_dims(probability_mat, axis=1)

    # Concatenate probability mat with itself to get a tensor with shape (2, pureStrategies_perPlayer, pureStrategies_perPlayer)
    probability_mat = K.concatenate([probability_mat] * playerNumber, axis=1)

    # Multiply the probability matrix by the game (payoffs) to get the expected payoffs for each player
    expectedPayoff_mat = game * probability_mat

    # Sum the expected payoff matrix for each player (eg: (Batch_Size, 2,3,3)->(Batch_Size, 2))
    payoffs = K.sum(expectedPayoff_mat, axis=[strategy_dim for strategy_dim in range(2, 2 + playerNumber)])

    return payoffs


# ********************************
def computePayoff(game, equilibrium, pureStrategies_perPlayer, *argv):
    """
    Function to compute the payoff each player gets with the input equilibrium and the input game (2 player games).
    """

    # Extract mix strategies of each player
    mixStrategies_p1 = tf.gather(equilibrium, 0, axis=1)[:, : pureStrategies_perPlayer[0]]
    mixStrategies_p1 = K.reshape(mixStrategies_p1, (tf.shape(mixStrategies_p1)[0], pureStrategies_perPlayer[0], 1))
    mixStrategies_p2 = tf.gather(equilibrium, 1, axis=1)[:, : pureStrategies_perPlayer[1]]
    mixStrategies_p2 = K.reshape(mixStrategies_p2, (tf.shape(mixStrategies_p2)[0], 1, pureStrategies_perPlayer[1]))

    # Multiply them together to get the probability matrix
    probability_mat = mixStrategies_p1 * mixStrategies_p2

    # Adding a new dimension
    probability_mat = K.expand_dims(probability_mat, axis=1)

    # Concatenate probability mat with itself to get a tensor with shape (2, pureStrategies_perPlayer, pureStrategies_perPlayer)
    probability_mat = K.concatenate([probability_mat, probability_mat], axis=1)

    # Multiply the probability matrix by the game (payoffs) to get the expected payoffs for each player
    expectedPayoff_mat = game * probability_mat

    # Sum the expected payoff matrix for each player (eg: (Batch_Size, 2,3,3)->(Batch_Size, 2))
    payoffs = K.sum(expectedPayoff_mat, axis=[2, 3])

    return payoffs


# ********************************
def computePayoff_2dBatch(game, equilibrium, pureStrategies_perPlayer, *argv):
    """
    Function to compute the payoff each player gets with the input equilibrium and the input game (2 player games)
    in a 2D batch setting.
    """

    # Extract mix strategies of each player
    mixStrategies_p1 = tf.gather(equilibrium, 0, axis=2)[:, :, : pureStrategies_perPlayer[0]]
    mixStrategies_p1 = K.reshape(mixStrategies_p1, (
        tf.shape(mixStrategies_p1)[0], tf.shape(mixStrategies_p1)[1], pureStrategies_perPlayer[0], 1))
    mixStrategies_p2 = tf.gather(equilibrium, 1, axis=2)[:, :, : pureStrategies_perPlayer[1]]
    mixStrategies_p2 = K.reshape(mixStrategies_p2, (
        tf.shape(mixStrategies_p2)[0], tf.shape(mixStrategies_p1)[1], 1, pureStrategies_perPlayer[1]))

    # Multiply them together to get the probability matrix
    probability_mat = mixStrategies_p1 * mixStrategies_p2

    # Adding a new dimension
    probability_mat = K.expand_dims(probability_mat, axis=2)

    # Concatenate probability mat with itself to get a tensor with shape (2, pureStrategies_perPlayer, pureStrategies_perPlayer)
    probability_mat = K.concatenate([probability_mat, probability_mat], axis=2)

    # Clone the game tensor to match the size of the equilibrium tensor
    game = tf.tile(tf.expand_dims(game, axis=1), [1, tf.shape(probability_mat)[1], 1, 1, 1])

    # Multiply the probability matrix by the game (payoffs) to get the expected payoffs for each player
    expectedPayoff_mat = game * probability_mat

    # Sum the expected payoff matrix for each player (eg: (Batch_Size, 10,2,3,3)->(Batch_Size, 10,2))
    payoffs = K.sum(expectedPayoff_mat, axis=[3, 4])

    return payoffs


# ********************************
@tf.function
def computePayoff_np_2dBatch(game, equilibrium, pureStrategies_perPlayer, playerNumber):
    """
    Function to compute the payoff each player gets with the input equilibrium and the input game
    (games with more than 2 players) in a 2D batch setting.
    """

    # Extract mix strategies of each player
    mixStrategies_perPlayer = [tf.gather(equilibrium, pl, axis=2)[:, :, : pureStrategies_perPlayer[pl]] for pl in range(playerNumber)]
    playerProbShape_grid = tuple(np.ones(playerNumber) + np.identity(playerNumber) * (np.array(pureStrategies_perPlayer) - 1))
    playerProbShape = [tuple((tf.shape(mixStrategies_perPlayer[0])[0], tf.shape(mixStrategies_perPlayer[0])[1]) +
                             tuple(playerProbShape_grid[pl])) for pl in range(playerNumber)]
    mixStrategies_perPlayer = [K.reshape(mixStrategies_perPlayer[pl], playerProbShape[pl]) for pl in range(playerNumber)]

    # Multiply them together to get the probability matrix
    probability_mat = mixStrategies_perPlayer[0]
    for pl in range(1, playerNumber):
        probability_mat *= mixStrategies_perPlayer[pl]

    # Adding a new dimension
    probability_mat = K.expand_dims(probability_mat, axis=2)

    # Concatenate probability mat with itself to get a tensor with shape (2, pureStrategies_perPlayer, pureStrategies_perPlayer)
    probability_mat = K.concatenate([probability_mat] * playerNumber, axis=2)

    # Clone the game tensor to match the size of the equilibrium tensor
    game = tf.tile(tf.expand_dims(game, axis=1), [1, tf.shape(probability_mat)[1], 1] + [1] * playerNumber)

    # Multiply the probability matrix by the game (payoffs) to get the expected payoffs for each player
    expectedPayoff_mat = game * probability_mat

    # Sum the expected payoff matrix for each player (eg: (Batch_Size, 10,2,3,3)->(Batch_Size, 10,2))
    payoffs = K.sum(expectedPayoff_mat, axis=[strategy_dim for strategy_dim in range(3, 3 + playerNumber)])

    return payoffs


# ********************************
def PayoffLoss_metric(game, payoff_Eq_function, pureStrategies_perPlayer, computePayoff_function, num_players):
    """
    Function to compute payoff loss as a metric during model training and testing
    """

    def Payoff(nashEq_true, nashEq_pred):
        _, loss_payoff_MSE = payoff_Eq_function(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred,
                                                computePayoff_function, num_players)

        return loss_payoff_MSE

    return Payoff


# ********************************
def lossFunction_Eq_MSE(*argv):
    """
    Function to compute the loss by taking the minimum of the mean square error of equilibria.
    """

    def Eq_MSE(nashEq_true, nashEq_pred):
        return MSE(nashEq_true, nashEq_pred)

    return Eq_MSE


# ********************************
def payoff_Eq_MSE(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred, computePayoff_function, num_players):
    """
    Function to compute the the mean square error of equilibria and the mean square error of payoffs resulted from the
    associated equilibria.
    This is not a loss function. It is used by other loss functions to compute their final loss values.
    """

    # Compute error
    error = nashEq_true - nashEq_pred

    # Compute the weight for the final result to recompense the replacement of nans with zeros and its effect
    # on the averaging
    nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(nashEq_true[0][0]), tf.int32))
    eq_n_elements = tf.size(nashEq_true[0][0])
    compensation_factor = tf.cast(eq_n_elements / (eq_n_elements - nan_count), tf.float32)

    # Replace nan values with 0
    error = tf.where(tf.math.is_nan(error), tf.zeros_like(error), error)

    # Computing the minimum of mean of squared error (MSE) of nash equilibria
    MSE_eq = K.mean(K.square(error), axis=[2, 3])
    min_index = K.argmin(MSE_eq, axis=1)
    loss_Eq_MSE = tf.gather_nd(MSE_eq,
                               tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype='int32')), axis=1))
    loss_Eq_MSE *= compensation_factor

    # Computing the payoffs given the selected output for each sample in the batch
    selected_trueNash = tf.gather_nd(nashEq_true,
                                     tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype='int32')),
                                              axis=1))
    payoff_true = computePayoff_function['computePayoff'](game, selected_trueNash, pureStrategies_perPlayer,
                                                          num_players)
    payoff_pred = computePayoff_function['computePayoff'](game, tf.gather(nashEq_pred, 0, axis=1),
                                                          pureStrategies_perPlayer, num_players)

    # Computing the mean squared error (MSE) of payoffs
    loss_payoff_MSE = K.mean(K.square(payoff_true - payoff_pred), axis=1)

    return loss_Eq_MSE, loss_payoff_MSE


# ********************************
def payoffv2_Eq_MSE(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred, computePayoff_function, num_players):
    """
    Function to compute the the mean square error of equilibria and the mean square error of payoffs resulted from
    the equilibria.
    This is not a loss function. It is used by other loss functions to compute their final loss values.
    """

    # Compute payoffs given the equilibria
    payoff_true = computePayoff_function['computePayoff_2dBatch'](game, nashEq_true, pureStrategies_perPlayer,
                                                                  num_players)
    payoff_pred = computePayoff_function['computePayoff'](game, tf.gather(nashEq_pred, 0, axis=1),
                                                          pureStrategies_perPlayer, num_players)
    payoff_pred = tf.tile(tf.expand_dims(payoff_pred, axis=1), [1, tf.shape(nashEq_true)[1], 1])

    # Compute MSE of payoffs
    loss_payoff_MSE = K.mean(K.min(K.mean(K.square(payoff_true - payoff_pred), axis=2), axis=1))

    # Compute MSE of equilibria
    loss_Eq_MSE = MSE(nashEq_true, nashEq_pred)

    return loss_Eq_MSE, loss_payoff_MSE


# ********************************
def lossFunction_payoff_MSE(game, payoff_Eq_function, payoffToEq_weight, pureStrategies_perPlayer,
                            computePayoff_function, num_players):
    """
    Function to compute the loss by taking the mean square error of payoffs resulted from the equilibria.
    """

    def payoff_MSE(nashEq_true, nashEq_pred):
        # Call th helper function to compute the MSE of payoffs
        _, loss = payoff_Eq_function(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred, computePayoff_function,
                                     num_players)

        # Averaging over the batch and return
        return K.mean(loss + (0 * _))

    return payoff_MSE


# ********************************
def lossFunction_payoff_Eq_weightedSum(game, payoff_Eq_function, payoffToEq_weight, pureStrategies_perPlayer,
                                       computePayoff_function, num_players):
    """
    Function to compute the loss by taking the weighted sum of MSE of equilibria and MSE of payoffs resulted
    from the equilibria.
    """

    def payoff_Eq_weightedSum(nashEq_true, nashEq_pred):
        # Call th helper function to compute the MSE of payoffs and equilibria
        loss_Eq_MSE, loss_payoff_MSE = payoff_Eq_function(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred,
                                                          computePayoff_function, num_players)

        # Compute the loss, average over the batch, and return
        return K.mean(loss_Eq_MSE + payoffToEq_weight * loss_payoff_MSE) / (1 + payoffToEq_weight)

    return payoff_Eq_weightedSum


# ********************************
def lossFunction_payoff_Eq_multiplication(game, payoff_Eq_function, payoffToEq_weight, pureStrategies_perPlayer,
                                          computePayoff_function, num_players):
    """
    Function to compute the loss by taking the multiplication of MSE of equilibria and hyperbolic tangent of MSE of
    payoffs resulted from the equilibria.
    """

    def payoff_Eq_multiplication(nashEq_true, nashEq_pred):
        # Call th helper function to compute the MSE of payoffs and equilibria
        loss_Eq_MSE, loss_payoff_MSE = payoff_Eq_function(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred,
                                                          computePayoff_function, num_players)

        # Compute the loss, average over the batch, and return
        return K.mean(2 * loss_Eq_MSE * loss_payoff_MSE) / (loss_Eq_MSE + loss_payoff_MSE)

    return payoff_Eq_multiplication


# ********************************
def hydra_lossFunction_Eq_MSE(*argv):
    """
    Function to compute the loss by taking maximum of the minimum (max-min) of the mean square error of equilibria.
    """

    def Eq_MSE(nashEq_true, nashEq_pred):
        return hydra_MSE(nashEq_true, nashEq_pred)

    return Eq_MSE


# ********************************
def hydra_payoff_Eq_MSE(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred, computePayoff_function, num_players):
    """
    Function to compute the maximum of minimum (max-min) of the mean square error of equilibria and the mean square
    error of payoffs resulted from the associated equilibria.
    This is not a loss function. It is used by other loss functions to compute their final loss values.
    """

    # Run the matching with the true and predicted equilibria as proposer and proposed (in terms of the Deferred Acceptance Algorithm terms) each time
    eq_MSE_trueProposer, payoff_MSE_trueProposer = hydra_oneSided_payoff_Eq_MSE(nashEq_true, nashEq_pred, game,
                                                                                pureStrategies_perPlayer,
                                                                                computePayoff_function, num_players)
    eq_MSE_trueProposed, payoff_MSE_trueProposed = hydra_oneSided_payoff_Eq_MSE(nashEq_pred, nashEq_true, game,
                                                                                pureStrategies_perPlayer,
                                                                                computePayoff_function, num_players)

    loss_Eq_MSE = K.mean((eq_MSE_trueProposer + eq_MSE_trueProposed) / 2.0)

    loss_payoff_MSE = K.mean((payoff_MSE_trueProposer + payoff_MSE_trueProposed) / 2.0)

    return loss_Eq_MSE, loss_payoff_MSE


# ********************************
def hydra_oneSided_payoff_Eq_MSE(nashEq_proposer, nashEq_proposed, game, pureStrategies_perPlayer,
                                 computePayoff_function, num_players):
    """
    Function to compute MSE of equilibria and payoffs from the matched equilibria with a proposer and proposed defined
    (in Deferred Acceptance Algorithm terms).
    """

    # Create a row-wise meshgrid of proposed equilibria for each sample in the batch by adding a new dimension and
    # replicate the array along that
    proposed_grid = tf.tile(tf.expand_dims(nashEq_proposed, axis=2), [1, 1, tf.shape(nashEq_proposed)[1], 1, 1])

    # Create a column-wise meshgrid of proposer equilibria for each sample in the batch by adding a new dimension and
    # replicate the array along that
    proposer_grid = tf.tile(tf.expand_dims(nashEq_proposer, axis=1), [1, tf.shape(nashEq_proposer)[1], 1, 1, 1])

    # Compute the weight for the final result to recompense the replacement of nans with zeros and its effect
    # on the averaging
    nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(nashEq_proposer[0][0] + nashEq_proposed[0][0]), tf.int32))
    eq_n_elements = tf.size(nashEq_proposer[0][0])
    compensation_factor = tf.cast(eq_n_elements / (eq_n_elements - nan_count), tf.float32)

    # Compute error grid
    error_grid = proposed_grid - proposer_grid

    # Replace nan values with 0
    error_grid = tf.where(tf.math.is_nan(error_grid), tf.zeros_like(error_grid), error_grid)

    # Computing indices of the minimum of mean of squared error (MSE) of nash equilibria
    MSE_eq = K.mean(K.square(error_grid), axis=[3, 4])
    min_index = K.argmin(MSE_eq, axis=2)

    # Convert the indices tensor to make it usable for later tf.gather_nd operations
    indexGrid = tf.reshape(min_index, (tf.shape(min_index)[0] * tf.shape(min_index)[1], 1, 1, 1))

    # Find the minimum of mean of squared error (MSE) of nash equilibria
    loss_Eq_MSE = K.max(tf.squeeze(tf.gather_nd(MSE_eq, indexGrid, batch_dims=2), axis=[2]), axis=1)
    loss_Eq_MSE *= compensation_factor

    # Computing the payoffs given the selected output for each sample in the batch
    selected_proposerNash = tf.squeeze(tf.gather_nd(proposer_grid, indexGrid, batch_dims=2), axis=2)
    payoff_proposer = computePayoff_function['computePayoff_2dBatch'](game, selected_proposerNash,
                                                                      pureStrategies_perPlayer, num_players)
    payoff_proposed = computePayoff_function['computePayoff_2dBatch'](game, nashEq_proposed, pureStrategies_perPlayer,
                                                                      num_players)

    # Computing the mean squared error (MSE) of payoffs
    loss_payoff_MSE = K.max(K.mean(K.square(payoff_proposed - payoff_proposer), axis=2), axis=1)

    return loss_Eq_MSE, loss_payoff_MSE


# ********************************
def hydra_payoffv2_Eq_MSE(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred, computePayoff_function,
                          num_players):
    """
    Function to compute the maximum of minimum (max-min) of the mean square error of equilibria and the mean square
    error of payoffs resulted from the equilibria.
    This is not a loss function. It is used by other loss functions to compute their final loss values.
    """

    # Run the matching with the true and predicted equilibria as proposer and proposed (in terms of the
    # Deferred Acceptance Algorithm terms) each time
    eq_MSE_trueProposer, payoff_MSE_trueProposer = hydra_oneSided_payoffv2_Eq_MSE(
        nashEq_true, nashEq_pred, game, pureStrategies_perPlayer, computePayoff_function, num_players)
    eq_MSE_trueProposed, payoff_MSE_trueProposed = hydra_oneSided_payoffv2_Eq_MSE(
        nashEq_pred, nashEq_true, game, pureStrategies_perPlayer, computePayoff_function, num_players)

    loss_Eq_MSE = K.mean((eq_MSE_trueProposer + eq_MSE_trueProposed) / 2.0)
    loss_payoff_MSE = (payoff_MSE_trueProposer + payoff_MSE_trueProposed) / 2.0

    return loss_Eq_MSE, loss_payoff_MSE


# ********************************
def hydra_oneSided_payoffv2_Eq_MSE(nashEq_proposer, nashEq_proposed, game, pureStrategies_perPlayer,
                                   computePayoff_function, num_players):
    """
    Function to compute MSE of equilibria and payoffs with a proposer and proposed defined
    (in Deferred Acceptance Algorithm terms).
    """

    # Compute payoffs given the equilibria
    payoff_proposed = computePayoff_function['computePayoff_2dBatch'](game, nashEq_proposed, pureStrategies_perPlayer,
                                                                      num_players)
    payoff_proposer = computePayoff_function['computePayoff_2dBatch'](game, nashEq_proposer, pureStrategies_perPlayer,
                                                                      num_players)

    # Create a row-wise meshgrid of proposed payoffs for each sample in the batch by adding a new dimension and
    # replicate the array along that
    proposed_grid = tf.tile(tf.expand_dims(payoff_proposed, axis=2), [1, 1, tf.shape(payoff_proposed)[1], 1])

    # Create a column-wise meshgrid of proposer payoffs for each sample in the batch by adding a new dimension and
    # replicate the array along that
    proposer_grid = tf.tile(tf.expand_dims(payoff_proposer, axis=1), [1, tf.shape(payoff_proposer)[1], 1, 1])

    # Compute MSE of payoffs
    squared_error = K.square(proposed_grid - proposer_grid)
    loss_payoff_MSE = K.mean(K.max(K.min(K.mean(squared_error, axis=3), axis=2), axis=1))

    # Compute MSE of equilibria
    loss_Eq_MSE = hydra_MSE(nashEq_proposed, nashEq_proposer)

    return loss_Eq_MSE, loss_payoff_MSE


# ********************************
def build_monohead_model(num_players, pure_strategies_per_player, max_equilibria, optimizer, lossType, payoffLoss_type,
                monohead_common_layer_sizes, monohead_layer_sizes_per_player, enable_batchNormalization,
                payoffToEq_weight=None, compute_epsilon=False):
    """
    Function to create the neural network model of NashNet. It returns the model.
    """

    # Get input shape
    input_shape = tuple((num_players,) + tuple(pure_strategies_per_player))

    # Build input layer & flatten it (so we can connect to the fully connected (Dense) layers)
    input_layer = tf.keras.layers.Input(shape=input_shape)
    flattened_input = tf.keras.layers.Flatten()(input_layer)

    # Build dense layers
    layer_sizes = monohead_common_layer_sizes
    current_layer = flattened_input
    for size in layer_sizes:
        if enable_batchNormalization:
            current_layer = tf.keras.layers.Dense(size)(current_layer)
            current_layer = tf.keras.layers.BatchNormalization()(current_layer)
            current_layer = tf.keras.layers.Activation('relu')(current_layer)
        else:
            current_layer = tf.keras.layers.Dense(size, activation='relu')(current_layer)
    final_dense = current_layer

    # Create output for each player
    layer_sizes_per_player = monohead_layer_sizes_per_player
    last_layer_player = []
    for _ in range(num_players):
        current_layer = tf.keras.layers.Dense(layer_sizes_per_player[0], activation='relu')(final_dense)
        for size in layer_sizes_per_player[1:]:
            current_layer = tf.keras.layers.Dense(size, activation='relu')(current_layer)
        last_layer_player.append(tf.keras.layers.Dense(max(pure_strategies_per_player))(current_layer))

    # Create softmax layers
    softmax = [tf.keras.layers.Activation('softmax')(last_layer_player[0])]
    for playerCounter in range(1, num_players):
        softmax.append(tf.keras.layers.Activation('softmax')(last_layer_player[playerCounter]))

    # Create the output layer
    concatenated_output = tf.keras.layers.concatenate(softmax)
    replicated_output = tf.keras.layers.concatenate([concatenated_output for _ in range(max_equilibria)])
    output_layer = tf.keras.layers.Reshape((max_equilibria, num_players, max(pure_strategies_per_player)))(replicated_output)

    # Create a keras sequential model from this architecture
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Determine which loss function to use
    loss_function, payoffLoss_function, computePayoff_function = chooseLossFunction(lossType, payoffLoss_type,
                                                                                    num_players, enableHydra=False)

    # Create the list of metrics
    metrics_list = [MSE, PayoffLoss_metric(input_layer, payoffLoss_function, pure_strategies_per_player,
                                                  computePayoff_function, num_players)]
    if compute_epsilon:
        metrics_list += [epsilon_approx(input_layer, pure_strategies_per_player, computePayoff_function, num_players, False),
                         # outperform_eq(input_layer, pure_strategies_per_player, computePayoff_function, num_players, False),
                         max_epsilon]

    # Compile the model
    model.compile(experimental_run_tf_function=False,
                  loss=loss_function(input_layer, payoffLoss_function, payoffToEq_weight, pure_strategies_per_player,
                                     computePayoff_function, num_players),
                  optimizer=optimizer,
                  metrics=metrics_list
                  )

    # Return the created model
    return model


# ********************************
def build_hydra_model(num_players, pure_strategies_per_player, max_equilibria, optimizer, lossType, payoffLoss_type,
                      sawfish_common_layer_sizes, bull_necked_common_layer_sizes, sawfish_head_layer_sizes,
                      bull_necked_head_layer_sizes, hydra_layer_sizes_per_player, enable_batchNormalization,
                      hydra_shape, payoffToEq_weight=None, compute_epsilon=False):
    """
    Function to create the hydra neural network model of NashNet. It returns the model.
    """

    # Get input shape
    input_shape = tuple((num_players,) + tuple(pure_strategies_per_player))

    # Build input layer & flatten it (So we can connect to Dense)
    input_layer = tf.keras.layers.Input(shape=input_shape)
    flattened_input = tf.keras.layers.Flatten(input_shape=input_shape)(input_layer)

    # Decide the layer sizes
    if hydra_shape == 'bull_necked':
        common_layer_sizes = bull_necked_common_layer_sizes
        head_layer_sizes = bull_necked_head_layer_sizes
    elif hydra_shape == 'sawfish':
        common_layer_sizes = sawfish_common_layer_sizes
        head_layer_sizes = sawfish_head_layer_sizes
    else:
        raise Exception('\nThe hydra_shape is not valid.\n')

    # Build dense (fully connected) layers common among the neural heads        
    current_layer = flattened_input
    for size in common_layer_sizes:
        if enable_batchNormalization:
            current_layer = tf.keras.layers.Dense(size)(current_layer)
            current_layer = tf.keras.layers.BatchNormalization()(current_layer)
            current_layer = tf.keras.layers.Activation('relu')(current_layer)
        else:
            current_layer = tf.keras.layers.Dense(size, activation='relu')(current_layer)
    final_common_dense = current_layer

    # Build dense (fully connected) layers for each neural head
    final_dense = [None] * max_equilibria
    current_layer = final_common_dense
    for headCounter in range(max_equilibria):
        for size in head_layer_sizes:
            if enable_batchNormalization:
                current_layer = tf.keras.layers.Dense(size)(current_layer)
                current_layer = tf.keras.layers.BatchNormalization()(current_layer)
                current_layer = tf.keras.layers.Activation('relu')(current_layer)
            else:
                current_layer = tf.keras.layers.Dense(size, activation='relu')(current_layer)
        final_dense[headCounter] = current_layer

    # Create output for each player and each head
    layer_sizes_per_player = hydra_layer_sizes_per_player
    last_layer_player = [None] * max_equilibria
    for headCounter in range(max_equilibria):
        last_layer_player[headCounter] = []
        for _ in range(num_players):
            current_layer = tf.keras.layers.Dense(layer_sizes_per_player[0], activation='relu')(final_dense[headCounter])
            for size in layer_sizes_per_player[1:]:
                current_layer = tf.keras.layers.Dense(size, activation='relu')(current_layer)
            last_layer_player[headCounter].append(tf.keras.layers.Dense(max(pure_strategies_per_player))(current_layer))

    # Create softmax layers (since games have been normalized so all values are between 0 and 1)
    softmax = [None] * max_equilibria
    for headCounter in range(max_equilibria):
        softmax[headCounter] = [tf.keras.layers.Activation('softmax')(last_layer_player[headCounter][0])]
        for playerCounter in range(1, num_players):
            softmax[headCounter].append(
                tf.keras.layers.Activation('softmax')(last_layer_player[headCounter][playerCounter]))

    # Create the output layer
    head_output = [None] * max_equilibria
    for headCounter in range(max_equilibria):
        head_output[headCounter] = tf.keras.layers.concatenate(softmax[headCounter])

    concatenated_heads = tf.keras.layers.concatenate(head_output)
    output_layer = tf.keras.layers.Reshape((max_equilibria, num_players, max(pure_strategies_per_player)))(
        concatenated_heads)

    # Create a keras sequential model from this architecture
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Determine which loss function to use
    loss_function, payoffLoss_function, computePayoff_function = chooseLossFunction(lossType, payoffLoss_type,
                                                                                    num_players, enableHydra=True)

    # Create the list of metrics
    metrics_list = [hydra_MSE, PayoffLoss_metric(input_layer, payoffLoss_function, pure_strategies_per_player,
                                                        computePayoff_function, num_players)]
    if compute_epsilon:
        metrics_list += [
            epsilon_approx(input_layer, pure_strategies_per_player, computePayoff_function, num_players, True),
            # outperform_eq(input_layer, pure_strategies_per_player, computePayoff_function, num_players, True),
            max_epsilon]

    # Compile the model
    model.compile(experimental_run_tf_function=False,
                  loss=loss_function(input_layer, payoffLoss_function, payoffToEq_weight, pure_strategies_per_player,
                                     computePayoff_function, num_players),
                  optimizer=optimizer,
                  metrics=metrics_list
                  )

    # Return the created model
    return model


# ********************************
def chooseLossFunction(lossType, payoffLoss_type, num_players, enableHydra):
    """
    Function to choose a loss function based on the config file preferences
    """

    # Based on the number of players determine which payoff computation function to use (2-player or n-player)
    if num_players == 2:
        computePayoff_function = {'computePayoff_2dBatch': computePayoff_2dBatch, 'computePayoff': computePayoff}
    elif num_players > 2:
        computePayoff_function = {'computePayoff_2dBatch': computePayoff_np_2dBatch, 'computePayoff': computePayoff_np}
    else:
        raise Exception('\nNumber of players is less than 2.\n')

    # Choose the payoff loss function
    if payoffLoss_type == 'closestPayoff':
        if enableHydra:
            payoffLoss_function = hydra_payoffv2_Eq_MSE
        else:
            payoffLoss_function = payoffv2_Eq_MSE
    elif payoffLoss_type == 'payoff_of_closestEq':
        if enableHydra:
            payoffLoss_function = hydra_payoff_Eq_MSE
        else:
            payoffLoss_function = payoff_Eq_MSE
    else:
        raise Exception('\nThe payoff loss type ' + payoffLoss_type + ' is undefined.\n')

    # Choose the loss function to return
    if lossType == 'Eq_MSE':
        if enableHydra:
            return hydra_lossFunction_Eq_MSE, payoffLoss_function, computePayoff_function
        else:
            return lossFunction_Eq_MSE, payoffLoss_function, computePayoff_function
    elif lossType == 'payoff_MSE':
        return lossFunction_payoff_MSE, payoffLoss_function, computePayoff_function
    elif lossType == 'payoff_Eq_weightedSum':
        return lossFunction_payoff_Eq_weightedSum, payoffLoss_function, computePayoff_function
    elif lossType == 'payoff_Eq_multiplication':
        return lossFunction_payoff_Eq_multiplication, payoffLoss_function, computePayoff_function
    else:
        raise Exception('\nThe loss type ' + lossType + ' is undefined.\n')


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
def saveTestData(test_files, saved_test_files_list, num_players, num_strategies):
    """
    Function to save test data to reuse for evaluation purposes
    """

    address = './Datasets/' + str(num_players) + 'P/' + str(num_strategies[0])
    for strategy in num_strategies[1:]:
        address += 'x' + str(strategy)

    assert os.path.exists(address), 'The path ' + address + ' does not exist'

    address += '/Test_Data/'
    if not os.path.exists(address):
        os.mkdir(address)

    saved_test_files_list += '.npy'

    np.save(os.path.join(address, saved_test_files_list), test_files)


# ********************************
def loadTestData(saved_test_files_list, num_players, num_strategies):
    """
    Function to save test data to reuse for evaluation purposes
    """

    address = './Datasets/' + str(num_players) + 'P/' + str(num_strategies[0])
    for strategy in num_strategies[1:]:
        address += 'x' + str(strategy)
    address += '/Test_Data/'

    saved_test_files_list += '.npy'

    test_files = np.load(os.path.join(address + saved_test_files_list))

    return test_files


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


# ********************************
@tf.function
def hydra_epsilon_equilibrium(nashEq_predicted, nashEq_true, game, pureStrategies_perPlayer,
                                 computePayoff_function, num_players):
    """
    Function to compute the epsilon in an epsilon-equilibrium setting for the hydra model.
    """

    # Create a row-wise meshgrid of predicted equilibria for each sample in the batch by adding a new dimension and
    # replicate the array along that
    predicted_grid = tf.tile(tf.expand_dims(nashEq_predicted, axis=2), [1, 1, tf.shape(nashEq_predicted)[1], 1, 1])

    # Create a column-wise meshgrid of true equilibria for each sample in the batch by adding a new dimension and
    # replicate the array along that
    true_grid = tf.tile(tf.expand_dims(nashEq_true, axis=1), [1, tf.shape(nashEq_true)[1], 1, 1, 1])

    # Compute error grid
    error_grid = predicted_grid - true_grid

    # Replace nan values with 0
    error_grid = tf.where(tf.math.is_nan(error_grid), tf.zeros_like(error_grid), error_grid)

    # Computing indices of the minimum of mean of squared error (MSE) of nash equilibria
    MSE_eq = K.mean(K.square(error_grid), axis=[3, 4])
    min_index = K.argmin(MSE_eq, axis=2)

    # Convert the indices tensor to make it usable for later tf.gather_nd operations
    indexGrid = tf.reshape(min_index, (tf.shape(min_index)[0] * tf.shape(min_index)[1], 1, 1, 1))

    # Find the matching true output for each sample in the batch
    selected_trueNash = tf.squeeze(tf.gather_nd(true_grid, indexGrid, batch_dims=2), axis=2)

    # Compute payoff of true equilibria (shape: [batch, max_eq, players])
    payoff_true = computePayoff_function['computePayoff_2dBatch'](game, selected_trueNash, pureStrategies_perPlayer,
                                                                  num_players)

    # Compute the difference in payoff a player when their strategy in true equilibrium is replaced by the predicted
    # strategy (shape: A list with size of players with each element [batch])
    epsilon_per_player = []
    outperform_eq_per_player = []
    for player in range(num_players):
        # Replace the prediction for a player on the true equilibrium
        unstacked_list = tf.unstack(selected_trueNash, axis=2, num=num_players)
        unstacked_list[player] = nashEq_predicted[:, :, player]
        approx_on_true = tf.stack(unstacked_list, axis=2)

        # Compute payoff for the modified equilibria
        approx_payoff_current_player = computePayoff_function['computePayoff_2dBatch'](game, approx_on_true, pureStrategies_perPlayer, num_players)

        # Compute epsilon and possible payoff improvement
        epsilon_per_player.append(tf.reduce_max(tf.maximum(payoff_true[:, :, player] - approx_payoff_current_player[:, :, player], 0), axis=1))
        outperform_eq_per_player.append(tf.reduce_all(tf.math.greater(approx_payoff_current_player[:, :, player], payoff_true[:, :, player]), axis=1))

    # Find the maximum epsilon for all players
    epsilon_stacked = tf.stack(epsilon_per_player, axis=1)
    epsilon = tf.reduce_max(epsilon_stacked)

    # Also find if any equilibrium better than the classical methods (true equilibrium) found
    outperform_stacked = tf.stack(outperform_eq_per_player, axis=1)
    outperform_no = tf.math.count_nonzero(tf.reduce_all(outperform_stacked, axis=1))

    return epsilon, outperform_no


# ********************************
@tf.function
def epsilon_equilibrium(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred, computePayoff_function, num_players):
    """
    Function to compute the epsilon in an epsilon-equilibrium setting.
    """

    # Compute error
    error = nashEq_true - nashEq_pred

    # Replace nan values with 0
    error = tf.where(tf.math.is_nan(error), tf.zeros_like(error), error)

    # Computing indices of the minimum of mean of squared error (MSE) of nash equilibria
    MSE_eq = K.mean(K.square(error), axis=[2, 3])
    min_index = K.argmin(MSE_eq, axis=1)

    # Find the matching true output for each sample in the batch
    selected_trueNash = tf.gather_nd(nashEq_true, tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype='int32')), axis=1))

    # Compute payoff of true equilibria (shape: [batch, players])
    payoff_true = computePayoff_function['computePayoff'](game, selected_trueNash, pureStrategies_perPlayer,
                                                          num_players)

    # Compute the difference in payoff a player when their strategy in true equilibrium is replaced by the predicted
    # strategy (shape: A list with size of players with each element [batch])
    epsilon_per_player = []
    outperform_eq_per_player = []
    for player in range(num_players):
        # Replace the prediction for a player on the true equilibrium
        unstacked_list = tf.unstack(selected_trueNash, axis=1, num=num_players)
        unstacked_list[player] = nashEq_pred[:, 0, player]
        approx_on_true = tf.stack(unstacked_list, axis=1)

        # Compute payoff for the modified equilibrium
        approx_payoff_current_player = computePayoff_function['computePayoff'](game, approx_on_true, pureStrategies_perPlayer, num_players)

        # Compute epsilon and possible payoff improvement
        epsilon_per_player.append(tf.maximum(payoff_true[:, player] - approx_payoff_current_player[:, player], 0))
        outperform_eq_per_player.append(tf.math.greater(approx_payoff_current_player[:, player], payoff_true[:, player]))

    # Find the maximum epsilon for all players
    epsilon_stacked = tf.stack(epsilon_per_player, axis=1)
    epsilon = tf.reduce_max(epsilon_stacked)

    # Also find if any equilibrium better than the classical methods (true equilibrium) found
    outperform_stacked = tf.stack(outperform_eq_per_player, axis=1)
    outperform_no = tf.math.count_nonzero(tf.reduce_all(outperform_stacked, axis=1))

    return epsilon, outperform_no


# ********************************
def epsilon_approx(game, pureStrategies_perPlayer, computePayoff_function, num_players, hydra_enabled):
    """
    Function to find epsilon in an epsilon-equilibrium in the context of approximate Nash equilibrium
    """

    if hydra_enabled:
        def epsilon(nashEq_true, nashEq_predicted):
            epsilon_, _ = hydra_epsilon_equilibrium(nashEq_predicted, nashEq_true, game, pureStrategies_perPlayer, computePayoff_function, num_players)
            return epsilon_
    else:
        def epsilon(nashEq_true, nashEq_predicted):
            epsilon_, _ = epsilon_equilibrium(game, pureStrategies_perPlayer, nashEq_true, nashEq_predicted, computePayoff_function, num_players)
            return epsilon_

    return epsilon


# ********************************
def outperform_eq(game, pureStrategies_perPlayer, computePayoff_function, num_players, hydra_enabled):
    """
    Function to find number of times predicted values for each player outperforms equilibrium
    """

    if hydra_enabled:
        def outperform_no(nashEq_true, nashEq_predicted):
            _, outperform_no_ = hydra_epsilon_equilibrium(nashEq_predicted, nashEq_true, game, pureStrategies_perPlayer, computePayoff_function, num_players)
            return outperform_no_
    else:
        def outperform_no(nashEq_true, nashEq_predicted):
            _, outperform_no_ = epsilon_equilibrium(game, pureStrategies_perPlayer, nashEq_true, nashEq_predicted, computePayoff_function, num_players)
            return outperform_no_

    return outperform_no


# ********************************
class EpsilonCallback(tf.keras.callbacks.Callback):
    """
    Class to save model and change learning rate during the training
    """
    
    def __init__(self):
        # Learning Rate Scheduler Variables
        self.epsilon = 0
        self.logs = None

    def on_batch_end(self, batch, logs={}):
        self.epsilon = tf.maximum(logs.get('epsilon'), self.epsilon)
        logs['max_epsilon'] = self.epsilon

    def on_epoch_end(self, epoch, logs=None):
        logs['max_epsilon'] = self.epsilon
        logs.pop("val_max_epsilon", None)

    def on_test_batch_end(self, batch, logs=None):
        self.epsilon = tf.maximum(logs.get('epsilon'), self.epsilon)
        logs['max_epsilon'] = self.epsilon
        self.logs = logs

    def on_test_end(self, logs=None):
        self.logs['max_epsilon'] = self.epsilon


def max_epsilon(*argv):
    return 0


# ********************************
def commutativity_test(test_data_generator, model, permutation_number):
    """
    A function to gauge the commutativity property of the model.
    :param test_data_generator: A generator of test games
    :param model: The trained model
    :param permutation_number: Number of random permutations to try
    :param test_batch_size: Batch size of test data
    :return: Returns the average of mean absolute errors after permutations in the players' strategies
    """

    # Check if at least one permutation is requested
    assert permutation_number > 0, 'Number of permutations for commutativity test must be more than zero.'

    # Get the number of batches in test data
    number_of_batches = test_data_generator.__len__()

    # Get an initial test sample batch
    tests_games, test_eq = test_data_generator.__getitem__(0)

    # Find the maximum possible permutations
    max_perm = 1
    for player_strategies in tests_games.shape[2:]:
        max_perm *= math.factorial(player_strategies)
    if permutation_number > max_perm:
        note_string = '\nWarning: ' + str(permutation_number) + ' permutations requested, but maximum number of unique permutations is ' + str(max_perm)
        permutation_number = max_perm
    else:
        note_string = ''

    # Print the starting message
    print('Starting the commutativity test. Number of random permutations: ' + str(permutation_number) + note_string)

    # Compute the weight to compensate for the zeros replaced in place of nan values
    nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(test_eq[0][0]), tf.int32))
    eq_n_elements = tf.size(test_eq[0][0])
    compensation_factor = tf.cast(eq_n_elements / (eq_n_elements - nan_count), tf.float32)

    # Test different permutations
    average_mae = 0
    for permute_no in range(permutation_number):
        # Print a message
        print('Starting permutation ' + str(permute_no + 1))

        total_sample_no = 0
        mae = 0
        for current_batch in range(number_of_batches):
            # Read a batch of test data
            tests_games, test_eq = test_data_generator.__getitem__(current_batch)

            # Save the original unpermuted games
            original_tests_games = tests_games.copy()

            # Permute the strategies for each player
            for idx, pl_strategies in enumerate(original_tests_games.shape[2:]):
                indices = tuple(np.random.permutation(pl_strategies))
                tests_games = np.take(tests_games, indices=indices, axis=(idx + 2))

            # Predict the equilibrium for the original games
            original_eq = model.predict(original_tests_games)

            # Save number of samples in the current batch
            sample_number = original_tests_games.shape[0]

            # Predict the permuted game
            new_eq = model.predict(tests_games)

            # Compute the mean absolute difference
            absolute_error = tf.math.abs(original_eq - new_eq)

            # Replace the dummy outputs (in the case of asymmetric games) with zero
            absolute_error = tf.where(tf.math.is_nan(test_eq), tf.zeros_like(absolute_error), absolute_error)

            # Compute the average of absolute errors
            mae += tf.reduce_mean(absolute_error) * compensation_factor * sample_number

            # Compute the total number of samples
            total_sample_no += sample_number

        mae /= total_sample_no
        average_mae += mae

    average_mae /= permutation_number

    return tf.get_static_value(average_mae)
