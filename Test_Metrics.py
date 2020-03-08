import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import math
from NashNet_utils import chooseLossFunction


# ********************************
@tf.function
def hydra_epsilon_equilibrium(nashEq_predicted, game, pureStrategies_perPlayer, computePayoff_function, num_players):
    """
    Function to compute the epsilon in an epsilon-equilibrium setting for the hydra model.
    """

    # Compute payoff of the predicted equilibria (shape: [batch, max_eq, players])
    payoff_pred = computePayoff_function['computePayoff_2dBatch'](game, nashEq_predicted, pureStrategies_perPlayer, num_players)

    # Unstack the predicted equilibria for each player
    unstacked_predicted_eq = tf.unstack(nashEq_predicted, axis=2, num=num_players)

    # Iterate over all players
    max_regret = 0.0
    for player in range(num_players):

        player_max_regret = 0.0

        # Iterate over all the pure strategies of a player
        for strategy in range(pureStrategies_perPlayer[player]):

            # Replace the prediction for the current player with a one-hot pure strategy corresponding to the current strategy
            pure_strategy_profile = tf.zeros_like(unstacked_predicted_eq[player])
            pure_strategy_profile = tf.unstack(pure_strategy_profile, axis=2)
            pure_strategy_profile[strategy] = tf.ones_like(pure_strategy_profile[strategy])
            pure_strategy_profile = tf.stack(pure_strategy_profile, axis=2)
            deviated_strategy = unstacked_predicted_eq.copy()
            deviated_strategy[player] = pure_strategy_profile
            pure_play = tf.stack(deviated_strategy, axis=2)

            # Compute payoff of the predicted equilibria for other players and the current strategy for the current player (shape: [batch, max_eq, players])
            payoff = computePayoff_function['computePayoff_2dBatch'](game, pure_play, pureStrategies_perPlayer, num_players)

            # Compute the payoff difference for the current player
            payoff_regret = payoff[:, :, player] - payoff_pred[:, :, player]
            payoff_regret = tf.maximum(payoff_regret, 0)

            # Save the maximum payoff regret for the current player
            player_max_regret = tf.maximum(player_max_regret, payoff_regret)

        # Save the maximum regret for all players
        max_regret = tf.maximum(max_regret, player_max_regret)

    # Set the epsilon (shape: [batch, max_eq])
    epsilon = max_regret

    return epsilon


# ********************************
@tf.function
def epsilon_equilibrium(game, pureStrategies_perPlayer, nashEq_pred, computePayoff_function, num_players):
    """
    Function to compute the epsilon in an epsilon-equilibrium setting.
    """

    # Compute payoff of the predicted equilibria (shape: [batch, players])
    payoff_pred = computePayoff_function['computePayoff'](game, nashEq_pred, pureStrategies_perPlayer, num_players)

    # Unstack the predicted equilibria for each player
    unstacked_predicted_eq = tf.unstack(nashEq_pred, axis=1, num=num_players)

    # Iterate over all players
    max_regret = 0
    for player in range(num_players):

        player_max_regret = 0

        # Iterate over all the pure strategies of a player
        for strategy in range(pureStrategies_perPlayer[player]):

            # Replace the prediction for the current player with a one-hot pure strategy corresponding to the current strategy
            pure_strategy_profile = tf.zeros_like(unstacked_predicted_eq[player])
            pure_strategy_profile = tf.unstack(pure_strategy_profile, axis=1)
            pure_strategy_profile[strategy] = tf.ones_like(pure_strategy_profile[strategy])
            pure_strategy_profile = tf.stack(pure_strategy_profile, axis=1)
            deviated_strategy = unstacked_predicted_eq.copy()
            deviated_strategy[player] = pure_strategy_profile
            pure_play = tf.stack(deviated_strategy, axis=1)

            # Compute payoff of the predicted equilibria for other players and the current strategy for the current player (shape: [batch, players])
            payoff = computePayoff_function['computePayoff'](game, pure_play, pureStrategies_perPlayer, num_players)

            # Compute the payoff difference for the current player
            payoff_regret = payoff[:, player] - payoff_pred[:, player]
            payoff_regret = tf.maximum(payoff_regret, 0)

            # Save the maximum payoff regret for the current player
            player_max_regret = tf.maximum(player_max_regret, payoff_regret)

        # Save the maximum regret for all players
        max_regret = tf.maximum(max_regret, player_max_regret)

    # Set the epsilon (shape: [batch])
    epsilon = max_regret

    return epsilon


# ********************************
@tf.function
def hydra_delta_equilibrium(nashEq_predicted, nashEq_true, game, pureStrategies_perPlayer,
                                 computePayoff_function, num_players):
    """
    Function to compute the delta (regret of not playing equilibrium strategy while others play that) in a hydra network.
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
    delta_per_player = []
    outperform_eq_per_player = []
    for player in range(num_players):
        # Replace the prediction for a player on the true equilibrium
        unstacked_list = tf.unstack(selected_trueNash, axis=2, num=num_players)
        unstacked_list[player] = nashEq_predicted[:, :, player]
        approx_on_true = tf.stack(unstacked_list, axis=2)

        # Compute payoff for the modified equilibria
        approx_payoff_current_player = computePayoff_function['computePayoff_2dBatch'](game, approx_on_true, pureStrategies_perPlayer, num_players)

        # Compute delta and possible payoff improvement
        delta_per_player.append(tf.maximum(payoff_true[:, :, player] - approx_payoff_current_player[:, :, player], 0))
        # delta_per_player.append(tf.reduce_max(tf.maximum(payoff_true[:, :, player] - approx_payoff_current_player[:, :, player], 0), axis=1))
        outperform_eq_per_player.append(tf.reduce_all(tf.math.greater(approx_payoff_current_player[:, :, player], payoff_true[:, :, player]), axis=1))

    # Find the maximum delta for all players
    delta_stacked = tf.stack(delta_per_player, axis=2)
    # delta_stacked = tf.stack(delta_per_player, axis=1)
    delta = tf.reduce_max(delta_stacked, axis=2)
    # delta = tf.reduce_max(delta_stacked)

    # Also find if any equilibrium better than the classical methods (true equilibrium) found (A scalar)
    outperform_stacked = tf.stack(outperform_eq_per_player, axis=1)
    outperform_no = tf.math.count_nonzero(tf.reduce_all(outperform_stacked, axis=1))

    return delta, outperform_no


# ********************************
@tf.function
def delta_equilibrium(game, pureStrategies_perPlayer, nashEq_true, nashEq_pred, computePayoff_function, num_players):
    """
    Function to compute the delta (regret of not playing equilibrium strategy while others play that).
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
    delta_per_player = []
    outperform_eq_per_player = []
    for player in range(num_players):
        # Replace the prediction for a player on the true equilibrium
        unstacked_list = tf.unstack(selected_trueNash, axis=1, num=num_players)
        unstacked_list[player] = nashEq_pred[:, 0, player]
        approx_on_true = tf.stack(unstacked_list, axis=1)

        # Compute payoff for the modified equilibrium
        approx_payoff_current_player = computePayoff_function['computePayoff'](game, approx_on_true, pureStrategies_perPlayer, num_players)

        # Compute delta and possible payoff improvement
        delta_per_player.append(tf.maximum(payoff_true[:, player] - approx_payoff_current_player[:, player], 0))
        outperform_eq_per_player.append(tf.math.greater(approx_payoff_current_player[:, player], payoff_true[:, player]))

    # Find the maximum delta for all players
    delta_stacked = tf.stack(delta_per_player, axis=1)
    delta = tf.reduce_max(delta_stacked, axis=1)
    # delta = tf.reduce_max(delta_stacked)

    # Also find if any equilibrium better than the classical methods (true equilibrium) found (A scalar)
    outperform_stacked = tf.stack(outperform_eq_per_player, axis=1)
    outperform_no = tf.math.count_nonzero(tf.reduce_all(outperform_stacked, axis=1))

    return delta, outperform_no


# ********************************
def epsilon_approx(game, pureStrategies_perPlayer, computePayoff_function, num_players, hydra_enabled):
    """
    Function to find epsilon in an epsilon-equilibrium in the context of approximate Nash equilibrium
    """

    if hydra_enabled:
        def epsilon(nashEq_true, nashEq_predicted):
            epsilon_ = hydra_epsilon_equilibrium(nashEq_predicted, game, pureStrategies_perPlayer, computePayoff_function, num_players)
            return epsilon_
    else:
        def epsilon(nashEq_true, nashEq_predicted):
            epsilon_ = epsilon_equilibrium(game, pureStrategies_perPlayer, nashEq_predicted, computePayoff_function, num_players)
            return epsilon_

    return epsilon


# ********************************
def delta_approx(game, pureStrategies_perPlayer, computePayoff_function, num_players, hydra_enabled):
    """
    Function to find delta by selecting the proper function to call.
    """

    if hydra_enabled:
        def delta(nashEq_true, nashEq_predicted):
            delta_, _ = hydra_delta_equilibrium(nashEq_predicted, nashEq_true, game, pureStrategies_perPlayer, computePayoff_function, num_players)
            return delta_
    else:
        def delta(nashEq_true, nashEq_predicted):
            delta_, _ = delta_equilibrium(game, pureStrategies_perPlayer, nashEq_true, nashEq_predicted, computePayoff_function, num_players)
            return delta_

    return delta


# ********************************
def outperform_eq(game, pureStrategies_perPlayer, computePayoff_function, num_players, hydra_enabled):
    """
    Function to find number of times predicted values for each player outperforms equilibrium
    """

    if hydra_enabled:
        def outperform_no(nashEq_true, nashEq_predicted):
            _, outperform_no_ = hydra_delta_equilibrium(nashEq_predicted, nashEq_true, game, pureStrategies_perPlayer, computePayoff_function, num_players)
            return outperform_no_
    else:
        def outperform_no(nashEq_true, nashEq_predicted):
            _, outperform_no_ = delta_equilibrium(game, pureStrategies_perPlayer, nashEq_true, nashEq_predicted, computePayoff_function, num_players)
            return outperform_no_

    return outperform_no


# ********************************
def epsilon_test(test_data_generator, model, pureStrategies_perPlayer, num_players, lossType, payoffLoss_type, hydra_enabled):
    """
    Function to compute epsilon and delta
    """

    # Set max and mean epsilon and delta to zero
    max_epsilon = 0.0
    max_delta = 0.0
    mean_epsilon = 0.0
    mean_delta = 0.0

    # Get the number of batches in test data
    number_of_batches = test_data_generator.__len__()

    # Define a progress bar
    progress_bar = tf.keras.utils.Progbar(number_of_batches, stateful_metrics=['max_epsilon', 'max_delta'],
                                          width=30, interval=0.05)

    # Determine the right function to compute payoff
    _, _, computePayoff_function = chooseLossFunction(lossType, payoffLoss_type, num_players, hydra_enabled)

    for current_batch in range(number_of_batches):

        # Read a batch of test data
        test_games, test_eq = test_data_generator.__getitem__(current_batch)
        test_games = tf.cast(test_games, tf.float32)
        test_eq = tf.cast(test_eq, tf.float32)

        # Get the proper functions to compute epsilon and delta
        epsilon_func = epsilon_approx(test_games, pureStrategies_perPlayer, computePayoff_function, num_players, hydra_enabled)
        delta_func = delta_approx(test_games, pureStrategies_perPlayer, computePayoff_function, num_players, hydra_enabled)

        # Predict the Nash equilibria
        eq_pred = tf.cast(model.predict(test_games, batch_size=test_games.shape[0]), tf.float32)

        # Get epsilon and delta for any prediction (hydra shape: [batch, max_eq], monohead shape: [batch])
        epsilon = epsilon_func(test_eq, eq_pred)
        delta = delta_func(test_eq, eq_pred)

        # Find the maximum epsilon and delta in the current batch
        epsilon_batch_max = tf.reduce_max(epsilon)
        delta_batch_max = tf.reduce_max(delta)

        # Find the maximum epsilon and delta so far
        max_epsilon = tf.maximum(max_epsilon, epsilon_batch_max)
        max_delta = tf.maximum(max_delta, delta_batch_max)

        # Find the average epsilon and delta in the current batch
        epsilon_batch_mean = tf.reduce_mean(epsilon)
        delta_batch_mean = tf.reduce_mean(delta)

        # Add the current mean values to overall mean values
        mean_epsilon += epsilon_batch_mean
        mean_delta += delta_batch_mean

        # Add to the progress bar
        progress_bar.add(1, values=[('epsilon', epsilon_batch_mean), ('delta', delta_batch_mean), ('max_epsilon', max_epsilon), ('max_delta', max_delta)])

    # Divide the sum of means by the number of batches
    mean_epsilon /= number_of_batches
    mean_delta /= number_of_batches

    return K.get_value(max_epsilon), K.get_value(max_delta), K.get_value(mean_epsilon), K.get_value(mean_delta)


# ********************************
def commutativity_test(test_data_generator, model, permutation_number):
    """
    A function to gauge the commutativity property of the model.
    :param test_data_generator: A generator of test games
    :param model: The trained model
    :param permutation_number: Number of random permutations to try
    :return: Returns the average of mean absolute errors after permutations in the players' strategies
    """

    # Set the random number generator seed to get repeatable results each time
    np.random.seed(0)

    # Check if at least one permutation is requested
    assert permutation_number > 0, 'Number of permutations for commutativity test must be more than zero.'

    # Get the number of batches in test data
    number_of_batches = test_data_generator.__len__()

    # Get an initial test sample batch
    test_games, test_eq = test_data_generator.__getitem__(0)

    # Find the maximum possible permutations
    max_perm = 1
    for player_strategies in test_games.shape[2:]:
        max_perm *= math.factorial(player_strategies)
    if permutation_number > max_perm:
        note_string = '\nWarning: ' + str(permutation_number) + ' permutations requested, but maximum number of unique permutations is ' + str(max_perm)
        permutation_number = max_perm
    else:
        note_string = ''

    # Print the starting message
    print('Starting the commutativity test. Number of random permutations: ' + str(permutation_number) + note_string)

    # Define a progress bar
    progress_bar = tf.keras.utils.Progbar(number_of_batches * permutation_number,
                                          stateful_metrics=['permutation number'], width=30, interval=0.05)

    # Compute the weight to compensate for the zeros replaced in place of nan values
    nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(test_eq[0][0]), tf.int32))
    eq_n_elements = tf.size(test_eq[0][0])
    compensation_factor = tf.cast(eq_n_elements / (eq_n_elements - nan_count), tf.float32)

    # Test different permutations
    average_mae = 0
    indices = [None] * test_games.shape[1]
    for permute_no in range(permutation_number):

        total_sample_no = 0
        mae = 0
        for current_batch in range(number_of_batches):

            # Read a batch of test data
            test_games, test_eq = test_data_generator.__getitem__(current_batch)
            test_eq = tf.cast(tf.tile(tf.expand_dims(test_eq, axis=1), [1, tf.shape(test_eq)[1], 1, 1, 1]), tf.float32)

            # Save the original unpermuted games
            original_tests_games = test_games.copy()

            # Permute the strategies for each player
            for idx, pl_strategies in enumerate(original_tests_games.shape[2:]):
                indices[idx] = tuple(np.random.permutation(pl_strategies))
                test_games = np.take(test_games, indices=indices[idx], axis=(idx + 2))

            # Predict the equilibrium for the original games
            original_eq = model.predict(original_tests_games, batch_size=original_tests_games.shape[0])

            # Save number of samples in the current batch
            sample_number = original_tests_games.shape[0]

            # Predict the permuted game
            new_eq = model.predict(test_games, batch_size=test_games.shape[0])

            # Change the order of probabilities to account for shuffling in strategies in the input game
            for player in range(new_eq.shape[2]):
                new_eq[:, :, player, : original_tests_games.shape[2 + player]] = np.take(new_eq[:, :, player, : original_tests_games.shape[2 + player]], indices=indices[player], axis=2)

            # print("\n\n\n\nOriginal Game:\n", repr(original_tests_games[0]), "\n\nGame:\n", repr(test_games[0]),
            #       "\n\nTrue:\n",
            #       test_eq_orig[0], "\n\n@@Indeices:", indices, "\nNew:\n", new_eq[0], "\n\n\nOld:\n", original_eq[0])

            # Tile the predictions to find the min differences (correspondences) later
            original_eq = tf.tile(tf.expand_dims(original_eq, axis=2), [1, 1, tf.shape(original_eq)[1], 1, 1])
            new_eq = tf.tile(tf.expand_dims(new_eq, axis=1), [1, tf.shape(new_eq)[1], 1, 1, 1])

            # Compute the mean absolute difference
            absolute_error = tf.math.abs(original_eq - new_eq)

            # Replace the dummy outputs (in the case of asymmetric games) with zero
            absolute_error = tf.where(tf.math.is_nan(test_eq), tf.zeros_like(absolute_error), absolute_error)

            # Find the min differences (correspondences) in the predicted equilibria
            absolute_error = tf.reduce_min(tf.reduce_mean(absolute_error, axis=[3, 4]), axis=2)

            # Compute the average of absolute errors
            mae += tf.reduce_mean(absolute_error) * compensation_factor * sample_number

            # Compute the total number of samples
            total_sample_no += sample_number

            # Add to the progress bar
            progress_bar.add(1, values=[('permutation number', permute_no + 1), ('commutativity mae', tf.reduce_mean(absolute_error) * compensation_factor)])

        mae /= total_sample_no
        average_mae += mae

    average_mae /= permutation_number

    return tf.get_static_value(average_mae)
