import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import nashpy as nash
import math, random
import pandas as pd
from sklearn import cluster


# ********************************
def GetTrainingDataFromNPY(data_file, labels_file):
    """
    Function to load dataset from two .npy files
    """

    data = np.load(data_file)
    labels = np.load(labels_file)

    return data, labels


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

    return K.mean(K.min(K.mean(K.square(nashEq_true - nashEq_pred), axis=[2, 3]), axis=1))


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

    return K.max(K.min(K.mean(K.square(proposed_grid - proposer_grid), axis=[3, 4]), axis=2), axis=1)


# ********************************
@tf.function
def computePayoff_np(game, equilibrium, pureStrategies_perPlayer, playerNumber):
    """
    Function to compute the payoff each player gets with the input equilibrium and the input game (games with more than 2 players).
    """

    # Extract mix strategies of each player
    mixStrategies_perPlayer = [tf.gather(equilibrium, pl, axis=1) for pl in range(playerNumber)]
    playerProbShape_grid = tuple(np.ones(playerNumber) + np.identity(playerNumber) * (pureStrategies_perPlayer - 1))
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
    probability_mat = K.concatenate([probability_mat, probability_mat], axis=1)

    # Multiply the probability matrix by the game (payoffs) to get the expected payoffs for each player
    expectedPayoff_mat = game * probability_mat

    # Sum the expected payoff matrix for each player (eg: (Batch_Size, 2,3,3)->(Batch_Size, 2,1))
    payoffs = K.sum(expectedPayoff_mat, axis=[2, 3])

    return payoffs


# ********************************
def computePayoff(game, equilibrium, pureStrategies_perPlayer, *argv):
    """
    Function to compute the payoff each player gets with the input equilibrium and the input game (2 player games).
    """

    # Extract mix strategies of each player
    mixStrategies_p1 = tf.gather(equilibrium, 0, axis=1)
    mixStrategies_p1 = K.reshape(mixStrategies_p1, (tf.shape(mixStrategies_p1)[0], pureStrategies_perPlayer, 1))
    mixStrategies_p2 = tf.gather(equilibrium, 1, axis=1)
    mixStrategies_p2 = K.reshape(mixStrategies_p2, (tf.shape(mixStrategies_p2)[0], 1, pureStrategies_perPlayer))

    # Multiply them together to get the probability matrix
    probability_mat = mixStrategies_p1 * mixStrategies_p2

    # Adding a new dimension
    probability_mat = K.expand_dims(probability_mat, axis=1)

    # Concatenate probability mat with itself to get a tensor with shape (2, pureStrategies_perPlayer, pureStrategies_perPlayer)
    probability_mat = K.concatenate([probability_mat, probability_mat], axis=1)

    # Multiply the probability matrix by the game (payoffs) to get the expected payoffs for each player
    expectedPayoff_mat = game * probability_mat

    # Sum the expected payoff matrix for each player (eg: (Batch_Size, 2,3,3)->(Batch_Size, 2,1))
    payoffs = K.sum(expectedPayoff_mat, axis=[2, 3])

    return payoffs


# ********************************
def computePayoff_2dBatch(game, equilibrium, pureStrategies_perPlayer, *argv):
    """
    Function to compute the payoff each player gets with the input equilibrium and the input game (2 player games) in a 2D batch setting.
    """

    # Extract mix strategies of each player
    mixStrategies_p1 = tf.gather(equilibrium, 0, axis=2)
    mixStrategies_p1 = K.reshape(mixStrategies_p1, (
    tf.shape(mixStrategies_p1)[0], tf.shape(mixStrategies_p1)[1], pureStrategies_perPlayer, 1))
    mixStrategies_p2 = tf.gather(equilibrium, 1, axis=2)
    mixStrategies_p2 = K.reshape(mixStrategies_p2, (
    tf.shape(mixStrategies_p2)[0], tf.shape(mixStrategies_p1)[1], 1, pureStrategies_perPlayer))

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

    # Sum the expected payoff matrix for each player (eg: (Batch_Size, 2,3,3)->(Batch_Size, 2,1))
    payoffs = K.sum(expectedPayoff_mat, axis=[3, 4])

    return payoffs


# ********************************
@tf.function
def computePayoff_np_2dBatch(game, equilibrium, pureStrategies_perPlayer, playerNumber):
    """
    Function to compute the payoff each player gets with the input equilibrium and the input game (games with more than 2 players) in a 2D batch setting.
    """

    # Extract mix strategies of each player
    mixStrategies_perPlayer = [tf.gather(equilibrium, pl, axis=2) for pl in range(playerNumber)]
    playerProbShape_grid = tuple(np.ones(playerNumber) + np.identity(playerNumber) * (pureStrategies_perPlayer - 1))
    playerProbShape = [tuple((tf.shape(mixStrategies_perPlayer[0])[0], tf.shape(mixStrategies_perPlayer[0])[1]) + tuple(
        playerProbShape_grid[pl])) for pl in
                       range(playerNumber)]
    mixStrategies_perPlayer = [K.reshape(mixStrategies_perPlayer[pl], playerProbShape[pl]) for pl in
                               range(playerNumber)]

    # Multiply them together to get the probability matrix
    probability_mat = mixStrategies_perPlayer[0]
    for pl in range(1, playerNumber):
        probability_mat *= mixStrategies_perPlayer[pl]

    # Adding a new dimension
    probability_mat = K.expand_dims(probability_mat, axis=2)

    # Concatenate probability mat with itself to get a tensor with shape (2, pureStrategies_perPlayer, pureStrategies_perPlayer)
    probability_mat = K.concatenate([probability_mat, probability_mat], axis=2)

    # Clone the game tensor to match the size of the equilibrium tensor
    game = tf.tile(tf.expand_dims(game, axis=1), [1, tf.shape(probability_mat)[1], 1, 1, 1])

    # Multiply the probability matrix by the game (payoffs) to get the expected payoffs for each player
    expectedPayoff_mat = game * probability_mat

    # Sum the expected payoff matrix for each player (eg: (Batch_Size, 2,3,3)->(Batch_Size, 2,1))
    payoffs = K.sum(expectedPayoff_mat, axis=[3, 4])

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
    Function to compute the the mean square error of equilibria and the mean square error of payoffs resulted from the associated equilibria.
    This is not a loss function. It is used by other loss functions to compute their final loss values.
    """

    # Computing the minimum of mean of squared error (MSE) of nash equilibria
    MSE_eq = K.mean(K.square(nashEq_true - nashEq_pred), axis=[2, 3])
    min_index = K.argmin(MSE_eq, axis=1)
    loss_Eq_MSE = tf.gather_nd(MSE_eq,
                               tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype='int32')),
                                        axis=1))

    # Computing the payoffs given the selected output for each sample in the batch
    selected_trueNash = tf.gather_nd(nashEq_true,
                                     tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype='int32')),
                                              axis=1))
    payoff_true = computePayoff_function['computePayoff'](game, selected_trueNash, pureStrategies_perPlayer,
                                                          num_players)
    payoff_pred = computePayoff_function['computePayoff'](game, tf.gather(nashEq_pred, 0, axis=1),
                                                          pureStrategies_perPlayer, num_players)

    # Computing the mean sqaured error (MSE) of payoffs
    loss_payoff_MSE = K.mean(K.square(payoff_true - payoff_pred), axis=1)

    return loss_Eq_MSE, loss_payoff_MSE


# ********************************
def payoffv2_Eq_MSE(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred, computePayoff_function, num_players):
    """
    Function to compute the the mean square error of equilibria and the mean square error of payoffs resulted from the equilibria.
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
    Function to compute the loss by taking the weighted sum of MSE of equilibria and MSE of payoffs resulted from the equilibria.
    """

    def payoff_Eq_weightedSum(nashEq_true, nashEq_pred):
        # Call th helper function to compute the MSE of payoffs and equilibria
        loss_Eq_MSE, loss_payoff_MSE = payoff_Eq_function(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred,
                                                          computePayoff_function, num_players)

        # Compute the loss, average over the batch, and return
        return K.mean(loss_Eq_MSE + payoffToEq_weight * loss_payoff_MSE)

    return payoff_Eq_weightedSum


# ********************************
def lossFunction_payoff_Eq_multiplication(game, payoff_Eq_function, payoffToEq_weight, pureStrategies_perPlayer,
                                          computePayoff_function, num_players):
    """
    Function to compute the loss by taking the multiplication of MSE of equilibria and hyperbolic tangent of MSE of payoffs resulted from the equilibria.
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
    Function to compute the maximum of minimum (max-min) of the mean square error of equilibria and the mean square error of payoffs resulted from the associated equilibria.
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
    Function to compute MSE of equilibria and payoffs from the matched equilibria with a proposer and proposed defined (in Deferred Acceptance Algorithm terms).
    """

    # Create a row-wise meshgrid of proposed equilibria for each sample in the batch by adding a new dimension and replicate the array along that
    proposed_grid = tf.tile(tf.expand_dims(nashEq_proposed, axis=2), [1, 1, tf.shape(nashEq_proposed)[1], 1, 1])

    # Create a column-wise meshgrid of proposer equilibria for each sample in the batch by adding a new dimension and replicate the array along that
    proposer_grid = tf.tile(tf.expand_dims(nashEq_proposer, axis=1), [1, tf.shape(nashEq_proposer)[1], 1, 1, 1])

    # Computing indeces of the minimum of mean of squared error (MSE) of nash equilibria
    MSE_eq = K.mean(K.square(proposed_grid - proposer_grid), axis=[3, 4])
    min_index = K.argmin(MSE_eq, axis=2)

    # Convert the indices tensor to make it usable for later tf.gather_nd operations
    indexGrid = tf.reshape(min_index, (tf.shape(min_index)[0] * tf.shape(min_index)[1], 1, 1, 1))

    # Find the minimum of mean of squared error (MSE) of nash equilibria
    loss_Eq_MSE = K.max(tf.squeeze(tf.gather_nd(MSE_eq, indexGrid, batch_dims=2)), axis=1)

    # Computing the payoffs given the selected output for each sample in the batch
    selected_proposerNash = tf.squeeze(tf.gather_nd(proposer_grid, indexGrid, batch_dims=2))
    payoff_proposer = computePayoff_function['computePayoff_2dBatch'](game, selected_proposerNash,
                                                                      pureStrategies_perPlayer, num_players)
    payoff_proposed = computePayoff_function['computePayoff_2dBatch'](game, nashEq_proposed, pureStrategies_perPlayer,
                                                                      num_players)

    # Computing the mean sqaured error (MSE) of payoffs
    loss_payoff_MSE = K.max(K.mean(K.square(payoff_proposed - payoff_proposer), axis=2), axis=1)

    return loss_Eq_MSE, loss_payoff_MSE


# ********************************
def hydra_payoffv2_Eq_MSE(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred, computePayoff_function,
                          num_players):
    """
    Function to compute the maximum of minimum (max-min) of the mean square error of equilibria and the mean square error of payoffs resulted from the equilibria.
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
def hydra_oneSided_payoffv2_Eq_MSE(nashEq_proposer, nashEq_proposed, game, pureStrategies_perPlayer,
                                   computePayoff_function, num_players):
    """
    Function to compute MSE of equilibria and payoffs with a proposer and proposed defined (in Deferred Acceptance Algorithm terms).
    """

    # Compute payoffs given the equilibria
    payoff_proposed = computePayoff_function['computePayoff_2dBatch'](game, nashEq_proposed, pureStrategies_perPlayer,
                                                                      num_players)
    payoff_proposer = computePayoff_function['computePayoff_2dBatch'](game, nashEq_proposer, pureStrategies_perPlayer,
                                                                      num_players)

    # Create a row-wise meshgrid of proposed payoffs for each sample in the batch by adding a new dimension and replicate the array along that
    proposed_grid = tf.tile(tf.expand_dims(payoff_proposed, axis=2), [1, 1, tf.shape(payoff_proposed)[1], 1])

    # Create a column-wise meshgrid of proposer payoffs for each sample in the batch by adding a new dimension and replicate the array along that
    proposer_grid = tf.tile(tf.expand_dims(payoff_proposer, axis=1), [1, tf.shape(payoff_proposer)[1], 1, 1])

    # Compute MSE of payoffs
    loss_payoff_MSE = K.mean(K.max(K.min(K.mean(K.square(proposed_grid - proposer_grid), axis=3), axis=2), axis=1))

    # Compute MSE of equilibria
    loss_Eq_MSE = hydra_MSE(nashEq_proposed, nashEq_proposer)

    return loss_Eq_MSE, loss_payoff_MSE


# ********************************
def build_model(num_players, pure_strategies_per_player, max_equilibria, optimizer, lossType, payoffLoss_type,
                enable_batchNormalization, payoffToEq_weight=None):
    """
    Function to create the neural network model of NashNet. It returns the model.
    """

    # Get input shape
    input_shape = tuple((num_players,) + tuple(pure_strategies_per_player for _ in range(num_players)))

    # Build input layer & flatten it (so we can connect to the fully connected (Dense) layers)
    input_layer = tf.keras.layers.Input(shape=input_shape)
    flattened_input = tf.keras.layers.Flatten()(input_layer)

    # Build dense layers
    layer_sizes = [200, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 200, 100]
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
    layer_sizes_per_player = [50, 50, 30]
    last_layer_player = []
    for _ in range(num_players):
        current_layer = tf.keras.layers.Dense(layer_sizes_per_player[0])(final_dense)
        for size in layer_sizes_per_player[1:]:
            current_layer = tf.keras.layers.Dense(size)(current_layer)
        last_layer_player.append(tf.keras.layers.Dense(pure_strategies_per_player)(current_layer))

    # Create softmax layers
    softmax = [tf.keras.layers.Activation('softmax')(last_layer_player[0])]
    for playerCounter in range(1, num_players):
        softmax.append(tf.keras.layers.Activation('softmax')(last_layer_player[playerCounter]))

    # Create the output layer
    concatenated_output = tf.keras.layers.concatenate(softmax)
    replicated_output = tf.keras.layers.concatenate([concatenated_output for _ in range(max_equilibria)])
    output_layer = tf.keras.layers.Reshape((max_equilibria, num_players, pure_strategies_per_player))(replicated_output)

    # Create a keras sequential model from this architecture
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Determine which loss function to use
    loss_function, payoffLoss_function, computePayoff_function = chooseLossFunction(lossType, payoffLoss_type,
                                                                                    num_players, enableHydra=False)

    # Compile the model
    model.compile(experimental_run_tf_function=False,
                  loss=loss_function(input_layer, payoffLoss_function, payoffToEq_weight, pure_strategies_per_player,
                                     computePayoff_function, num_players),
                  optimizer=optimizer,
                  metrics=[MSE, PayoffLoss_metric(input_layer, payoffLoss_function, pure_strategies_per_player,
                                                  computePayoff_function, num_players)]
                  )

    # Return the created model
    return model


# ********************************
def build_hydra_model(num_players, pure_strategies_per_player, max_equilibria, optimizer, lossType, payoffLoss_type,
                      enable_batchNormalization, hydra_shape, payoffToEq_weight=None):
    """
    Function to create the hydra neural network model of NashNet. It returns the model.
    """

    # Get input shape
    input_shape = tuple((num_players,) + tuple(pure_strategies_per_player for _ in range(num_players)))

    # Build input layer & flatten it (So we can connect to Dense)
    input_layer = tf.keras.layers.Input(shape=input_shape)
    flattened_input = tf.keras.layers.Flatten(input_shape=input_shape)(input_layer)

    # Decide the layer sizes
    if hydra_shape == 'bull_necked':
        common_layer_sizes = [100, 200, 500, 500, 500, 500, 500, 500, 500, 500, 500, 200, 200, 200, 100, 100, 100, 100,
                              100]
        head_layer_sizes = []
    elif hydra_shape == 'sawfish':
        common_layer_sizes = [100, 200, 500, 500, 500, 500, 500, 500, 500, 500, 500, 200, 200, 200]
        head_layer_sizes = [100]
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
    layer_sizes_per_player = [50, 50, 30]
    last_layer_player = [None] * max_equilibria
    for headCounter in range(max_equilibria):
        last_layer_player[headCounter] = []
        current_layer = tf.keras.layers.Dense(layer_sizes_per_player[0])(final_dense[headCounter])
        for _ in range(num_players):
            for size in layer_sizes_per_player[1:]:
                current_layer = tf.keras.layers.Dense(size)(current_layer)
            last_layer_player[headCounter].append(tf.keras.layers.Dense(pure_strategies_per_player)(current_layer))

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
    output_layer = tf.keras.layers.Reshape((max_equilibria, num_players, pure_strategies_per_player))(
        concatenated_heads)

    # Create a keras sequential model from this architecture
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Determine which loss function to use
    loss_function, payoffLoss_function, computePayoff_function = chooseLossFunction(lossType, payoffLoss_type,
                                                                                    num_players, enableHydra=True)

    # Compile the model
    model.compile(experimental_run_tf_function=False,
                  loss=loss_function(input_layer, payoffLoss_function, payoffToEq_weight, pure_strategies_per_player,
                                     computePayoff_function, num_players),
                  optimizer=optimizer,
                  metrics=[hydra_MSE, PayoffLoss_metric(input_layer, payoffLoss_function, pure_strategies_per_player,
                                                        computePayoff_function, num_players)]
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
def generate_nash(game):
    """
    Function to compute Nash equilibrium based on the classical methods.
    """

    equilibrium = nash.Game(game[0], game[1])

    nash_support_enumeration = []
    nash_lemke_howson_enumeration = []
    nash_vertex_enumeration = []

    try:
        for eq in equilibrium.support_enumeration():
            nash_support_enumeration.append(eq)

        for eq in equilibrium.lemke_howson_enumeration():
            nash_lemke_howson_enumeration.append(eq)

        for eq in equilibrium.vertex_enumeration():
            nash_vertex_enumeration.append(eq)
    except:
        pass

    return nash_support_enumeration, nash_lemke_howson_enumeration, nash_vertex_enumeration


# ********************************
def saveHistory(trainingHistory, model, trainingHistory_file):
    """
    Function to save the training history and evaluation results in two separate files
    """

    # Save training history
    trainingHistory_dataFrame = pd.DataFrame(trainingHistory.history)
    trainingHistory_dataFrame.index += 1
    trainingHistory_dataFrame.to_csv('./Reports/' + trainingHistory_file)

    # Save evaluation results


#     pd.DataFrame([model.metrics_names, evaluationResults]).to_csv('./Reports/' + testResults_file, index = False)

# ********************************
def saveTestData(testSamples, testEqs, num_players, num_strategies):
    """
    Function to save test data to reuse for evaluation purposes
    """

    address = './Datasets/Test_Data/' + str(num_players) + 'P/' + str(num_strategies) + 'x' + str(num_strategies) + '/'

    np.save(address + 'Saved_Test_Games.npy', testSamples)
    np.save(address + 'Saved_Test_Equilibria.npy', testEqs)


# ********************************
def loadTestData(test_games_file, test_equilibria_file, max_equilibria, num_players, num_strategies):
    """
    Function to save test data to reuse for evaluation purposes
    """

    address = './Datasets/Test_Data/' + str(num_players) + 'P/' + str(num_strategies) + 'x' + str(num_strategies) + '/'
    testSamples, testEqs = GetTrainingDataFromNPY(address + test_games_file, address + test_equilibria_file)

    # Limit the number of true equilibria for each sample game if they are more than max_equilibria
    if max_equilibria < testEqs.shape[1]:
        testEqs = testEqs[:, 0: max_equilibria, :, :]
    elif max_equilibria > testEqs.shape[1]:
        raise Exception(
            '\nmax_equilibria is larger than the number of per sample true equilibria in the provided test dataset.\n')

    return testSamples, testEqs


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
def printExamples(numberOfExamples, testSamples, testEqs, nn_model, examples_print_file, pureStrategies_per_player,
                  lossType, payoffLoss_type, num_players, enable_hydra, cluster_examples, payoffToEq_weight=None):
    """
    Function to make some illustrative predictions and print them
    """

    # Check the requested number of examples is feasible
    if numberOfExamples > testSamples.shape[0]:
        print("\n\nNumber of example predictions more than the number of test samples\n")
        exit()
    elif numberOfExamples == 0:
        return

    # Fetching an example game from the test set
    randomExample = random.randint(0, testSamples.shape[0] - numberOfExamples)
    exampleGame = testSamples[randomExample: randomExample + numberOfExamples]
    nash_true = testEqs[randomExample: randomExample + numberOfExamples]
    nash_true = nash_true.astype('float32')

    # Predicting a Nash equilibrium for the example game
    nash_predicted = nn_model.predict(exampleGame).astype('float32')

    # Set the precision of float numbers
    np.set_printoptions(precision=4)

    # Determine which loss function to use
    lossFunction, payoffLoss_function, computePayoff_function = chooseLossFunction(lossType, payoffLoss_type,
                                                                                   num_players, enable_hydra)

    # Open file for writing the results into
    printFile = open('./Reports/' + examples_print_file, "w")

    for exampleCounter in range(numberOfExamples):
        # Computing the loss
        lossFunction_instance = lossFunction(exampleGame[exampleCounter], payoffLoss_function, payoffToEq_weight,
                                             pureStrategies_per_player, computePayoff_function, num_players)
        loss = lossFunction_instance(np.expand_dims(nash_true[exampleCounter], axis=0),
                                     np.expand_dims(nash_predicted[exampleCounter], axis=0))

        # Compute the Nash equilibrium for the current game to get only distinctive equilibria
        distinctive_NashEquilibria, _, _ = generate_nash(exampleGame[exampleCounter])

        # If enabled, cluster the predicted Nash equilibria
        if cluster_examples:
            predictedEq = clustering(np.reshape(nash_predicted[exampleCounter],
                                                (nash_predicted.shape[1], num_players * pureStrategies_per_player)),
                                     num_players, pureStrategies_per_player)
        else:
            predictedEq = nash_predicted[exampleCounter]

        # Printing the results for the example game
        listOfTrueEquilibria = [distinctive_NashEquilibria[i] for i in range(len(distinctive_NashEquilibria))]
        printString = ("\n______________\nExample {}:\nTrue: \n" + (
                    "{}\n" * len(distinctive_NashEquilibria)) + "\nPredicted: \n{}\n\nLoss: {}\n\n") \
            .format(*([exampleCounter + 1] + listOfTrueEquilibria + list([predictedEq]) + [K.get_value(loss)])).replace(
            "array", "")
        print(printString)

        # Write the string to the file
        printFile.write(printString)

    printFile.close()


# ********************************
class NashNet_Metrics(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, num_cycles, max_epochs, save_dir, save_name):
        # Learning Rate Scheduler Variables
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.num_cycles = num_cycles

        # Checkpoint and saving variables
        self.save_dir = save_dir
        self.save_name = save_name

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
        if (epoch % (self.max_epochs / self.num_cycles) == 0) and (epoch != 0):
            # Get snapshot number
            snpashot_num = int(epoch / int(self.max_epochs / self.num_cycles))

            # Save the weights
            self.model.save_weights(
                self.save_dir.rstrip('/') + "/" + self.save_name + '_snapshot' + str(snpashot_num) + '_epoch' + str(
                    epoch) + '.h5')


#     def on_batch_begin(self, batch, logs={}):
#         return

#     def on_batch_end(self, batch, logs={}):
#         return

# ********************************
def clustering(pred, num_players, num_strategies):
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
         range(clusterNumber)]), (clusterNumber, 2, 3))
