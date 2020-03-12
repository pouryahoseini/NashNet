import tensorflow as tf
from Loss import *


# ********************************
def build_monohead_model(num_players, pure_strategies_per_player, max_equilibria, optimizer, lossType, payoffLoss_type,
                monohead_common_layer_sizes, monohead_layer_sizes_per_player, enable_batchNormalization,
                payoffToEq_weight=None):
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
                      hydra_shape, payoffToEq_weight=None):
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

    # Compile the model
    model.compile(experimental_run_tf_function=False,
                  loss=loss_function(input_layer, payoffLoss_function, payoffToEq_weight, pure_strategies_per_player,
                                     computePayoff_function, num_players),
                  optimizer=optimizer,
                  metrics=metrics_list
                  )

    # Return the created model
    return model
