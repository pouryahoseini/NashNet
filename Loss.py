import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


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
def lossFunction_payoff_Eq_weightedAverage(game, payoff_Eq_function, payoffToEq_weight, pureStrategies_perPlayer,
                                       computePayoff_function, num_players):
    """
    Function to compute the loss by taking the weighted average of MSE of equilibria and MSE of payoffs resulted
    from the equilibria.
    """

    def payoff_Eq_weightedAverage(nashEq_true, nashEq_pred):
        # Call th helper function to compute the MSE of payoffs and equilibria
        loss_Eq_MSE, loss_payoff_MSE = payoff_Eq_function(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred,
                                                          computePayoff_function, num_players)

        # Compute the loss, average over the batch, and return
        return K.mean(loss_Eq_MSE + payoffToEq_weight * loss_payoff_MSE) / (1 + payoffToEq_weight)

    return payoff_Eq_weightedAverage


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
    elif lossType == 'payoff_Eq_weightedAverage':
        return lossFunction_payoff_Eq_weightedAverage, payoffLoss_function, computePayoff_function
    elif lossType == 'payoff_Eq_multiplication':
        return lossFunction_payoff_Eq_multiplication, payoffLoss_function, computePayoff_function
    else:
        raise Exception('\nThe loss type ' + lossType + ' is undefined.\n')
