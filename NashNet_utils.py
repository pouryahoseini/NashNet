
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import nashpy as nash


# ********************************
def GetTrainingDataFromNPY(data_file, labels_file):
    '''
    Function to load dataset from two .npy files
    '''
    data = np.load(data_file)
    labels = np.load(labels_file)

    return data, labels


# ********************************
def unisonShuffle(a, b):
    '''
    Function to shuffle two numpy arrays in unison
    '''

    assert a.shape[0] == b.shape[0]

    p = np.random.permutation(a.shape[0])

    return a[p], b[p]


# ********************************
def MSE(nashEq_true, nashEq_pred):
    '''
    Function to compute the correct mean squared error (mse) as a metric during model training and testing
    '''

    return K.mean(K.min(K.mean(K.square(nashEq_true - nashEq_pred), axis=[2, 3]), axis=1))


# This just moves mse to a creator function so that it's built into the graph properly when creating the model
def make_MSE():
    def MSE(nashEq_true, nashEq_pred):
        return K.mean(K.min(K.mean(K.square(nashEq_true - nashEq_pred), axis=[2, 3]), axis=1))

    return MSE


# ********************************
def computePayoff_np(game, equilibrium, pureStrategies_perPlayer, playerNumber):
    '''
    Function to compute the payoff each player gets with the input equilibrium and the input game (games with more than 2 players).
    '''

    # Extract mix strategies of each player
    mixStrategies_perPlayer = [tf.gather(equilibrium, 0, axis=1) for pl in range(playerNumber)]
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
def computePayoff(game, equilibrium, pureStrategies_perPlayer):
    '''
    Function to compute the payoff each player gets with the input equilibrium and the input game (2 player games).
    '''

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
def PayoffLoss_metric(game, weight_EqDiff, weight_payoffDiff, pureStrategies_perPlayer):
    '''
    Function to compute payoff loss as a metric during model training and testing
    '''

    def computePayoff1(equilibrium):
        '''
        Function to compute the payoff each player gets with the input equilibrium and the input game (2 player games).
        '''

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

    def PayoffLoss(nashEq_true, nashEq_pred):
        # Computing the minimum of sum of squared error (SSE) of nash equilibria
        SSE_eq = K.sum(K.square(nashEq_true - nashEq_pred), axis=[2, 3])
        min_index = K.argmin(SSE_eq, axis=1)

        # Computing the minimum of sum of sqaured error (SSE) of payoffs
        selected_trueNash = tf.gather_nd(nashEq_true, tf.stack(
            (tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype='int32')), axis=1))
        payoff_true = computePayoff1(selected_trueNash)
        payoff_pred = computePayoff1(tf.gather(nashEq_pred, 0, axis=1))

        # Find the difference between the predicted and true payoffs
        payoff_sqDiff = K.square(payoff_true - payoff_pred)
        loss_payoffs_SSE = K.mean(K.sum(payoff_sqDiff, axis=1))

        return loss_payoffs_SSE
    return PayoffLoss


# Creates a loss function to minimize difference in payoff
def create_sse_loss(args=dict()):
    def enclosed_loss(nash_true, nash_pred):
        # Compute the minimum of sum of squared error (SSE) of nash eq
        SSE = K.sum(K.square(nash_true - nash_pred), axis=[2, 3])
        # Find minimum error
        min_index = K.argmin(SSE, axis=1)
        # loss_SSE = tf.gather_nd(SSE,
        #                         tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype='int32')),
        #                                  axis=1))
        loss_SSE = tf.gather_nd(SSE,
                                tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.fill(tf.shape(min_index), 2)),
                                         axis=1))
        return loss_SSE
    return enclosed_loss

# Loss function for multiheaded shit
def create_hydra_sse_loss(args=dict()):
    def enclosed_loss(nash_true, nash_pred):
        # Compute the minimum of sum of squared error (SSE) of nash eq
        SSE = K.sum(K.square(nash_true - nash_pred), axis=[2, 3])
        # Find minimum error
        min_index = K.argmin(SSE, axis=1)
        # loss_SSE = tf.gather_nd(SSE,
        #                         tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.cast(min_index, dtype='int32')),
        #                                  axis=1))
        loss_SSE = tf.gather_nd(SSE,
                                tf.stack((tf.range(0, tf.shape(min_index)[0]), tf.fill(tf.shape(min_index), 2)),
                                         axis=1))
        return loss_SSE
    return enclosed_loss

def build_model(num_players: object, pure_strategies_per_player: object, max_equilibria: object, optimizer: object, metrics: object, create_loss: object) -> object:
    # Get input shape
    input_shape = tuple((num_players,) + tuple(pure_strategies_per_player for _ in range(num_players)))

    # Build input layer & flatten it (So we can connect to Dense)
    input_layer = tf.keras.layers.Input(shape=input_shape)
    flattened_input = tf.keras.layers.Flatten(input_shape=input_shape)(input_layer)

    # Build dense layers
    layer_sizes = [200, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 200, 100]
    prev_layer = flattened_input
    for size in layer_sizes:
        prev_layer = tf.keras.layers.Dense(size, activation='relu')(prev_layer)
    final_dense = prev_layer

    # Create output for each player
    last_layer_player = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))

    # Create softmax layers (Since games have been normalized so all values are between 0 and 1?)
    softmax = [tf.keras.layers.Activation('softmax')(last_layer_player[0])]
    for playerCounter in range(1, num_players):
        softmax.append(tf.keras.layers.Activation('softmax')(last_layer_player[playerCounter]))

    # Create the output layer
    concatenated_output = tf.keras.layers.concatenate([softmax[pl] for pl in range(num_players)])
    replicated_output = tf.keras.layers.concatenate([concatenated_output for i in range(max_equilibria)])
    output_layer = tf.keras.layers.Reshape((max_equilibria, num_players, pure_strategies_per_player))(replicated_output)

    # Create a keras sequential model from this architecture
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    for i, elem in enumerate(metrics):
        if elem == "payoff_loss_metric":
            metrics[i] = PayoffLoss_metric(input_layer, 1, 1, pure_strategies_per_player)
        elif elem == "mse_custom":
            metrics[i] = make_MSE()

    # Compile the model
    model.compile(loss=create_loss(), optimizer=optimizer, metrics=metrics)

    # Return the created model
    return model


def build_hydra_model(num_players: object, pure_strategies_per_player: object, max_equilibria: object, optimizer: object, metrics: object, create_loss: object) -> object:
    # Get input shape
    input_shape = tuple((num_players,) + tuple(pure_strategies_per_player for _ in range(num_players)))

    # Build input layer & flatten it (So we can connect to Dense)
    input_layer = tf.keras.layers.Input(shape=input_shape)
    flattened_input = tf.keras.layers.Flatten(input_shape=input_shape)(input_layer)

    # Build dense layers
    layer_sizes = [200, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 200, 100]
    prev_layer = flattened_input
    for size in layer_sizes:
        prev_layer = tf.keras.layers.Dense(size, activation='relu')(prev_layer)
    final_dense = prev_layer

    # Create output for each player and each head
    # THIS IS UGLY AND I HATE IT
    last_layer_player0 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player0.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))
    last_layer_player1 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player1.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))
    last_layer_player2 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player2.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))
    last_layer_player3 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player3.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))
    last_layer_player4 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player4.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))
    last_layer_player5 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player5.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))
    last_layer_player6 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player6.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))
    last_layer_player7 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player7.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))
    last_layer_player8 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player8.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))
    last_layer_player9 = [tf.keras.layers.Dense(pure_strategies_per_player)(final_dense)]
    for _ in range(1, num_players):
        last_layer_player9.append(tf.keras.layers.Dense(pure_strategies_per_player)(final_dense))

    # Create softmax layers (Since games have been normalized so all values are between 0 and 1?)
    softmax0 = [tf.keras.layers.Activation('softmax')(last_layer_player0[0])]
    for playerCounter in range(1, num_players):
        softmax0.append(tf.keras.layers.Activation('softmax')(last_layer_player0[playerCounter]))
    softmax1 = [tf.keras.layers.Activation('softmax')(last_layer_player1[0])]
    for playerCounter in range(1, num_players):
        softmax1.append(tf.keras.layers.Activation('softmax')(last_layer_player1[playerCounter]))
    softmax2 = [tf.keras.layers.Activation('softmax')(last_layer_player2[0])]
    for playerCounter in range(1, num_players):
        softmax2.append(tf.keras.layers.Activation('softmax')(last_layer_player2[playerCounter]))
    softmax3 = [tf.keras.layers.Activation('softmax')(last_layer_player3[0])]
    for playerCounter in range(1, num_players):
        softmax3.append(tf.keras.layers.Activation('softmax')(last_layer_player3[playerCounter]))
    softmax4 = [tf.keras.layers.Activation('softmax')(last_layer_player4[0])]
    for playerCounter in range(1, num_players):
        softmax4.append(tf.keras.layers.Activation('softmax')(last_layer_player4[playerCounter]))
    softmax5 = [tf.keras.layers.Activation('softmax')(last_layer_player5[0])]
    for playerCounter in range(1, num_players):
        softmax5.append(tf.keras.layers.Activation('softmax')(last_layer_player5[playerCounter]))
    softmax6 = [tf.keras.layers.Activation('softmax')(last_layer_player6[0])]
    for playerCounter in range(1, num_players):
        softmax6.append(tf.keras.layers.Activation('softmax')(last_layer_player6[playerCounter]))
    softmax7 = [tf.keras.layers.Activation('softmax')(last_layer_player7[0])]
    for playerCounter in range(1, num_players):
        softmax7.append(tf.keras.layers.Activation('softmax')(last_layer_player7[playerCounter]))
    softmax8 = [tf.keras.layers.Activation('softmax')(last_layer_player8[0])]
    for playerCounter in range(1, num_players):
        softmax8.append(tf.keras.layers.Activation('softmax')(last_layer_player8[playerCounter]))
    softmax9 = [tf.keras.layers.Activation('softmax')(last_layer_player9[0])]
    for playerCounter in range(1, num_players):
        softmax9.append(tf.keras.layers.Activation('softmax')(last_layer_player9[playerCounter]))

    # Create the output layer
    output_layer = tf.keras.layers.concatenate(softmax0, softmax1, softmax2, softmax3, softmax4,
                                               softmax5, softmax6, softmax7, softmax8, softmax9)
    # concatenated_output = tf.keras.layers.concatenate([softmax[pl] for pl in range(num_players)])
    # replicated_output = tf.keras.layers.concatenate([concatenated_output for i in range(max_equilibria)])
    # output_layer = tf.keras.layers.Reshape((max_equilibria, num_players, pure_strategies_per_player))(replicated_output)

    # Create a keras sequential model from this architecture
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    for i, elem in enumerate(metrics):
        if elem == "payoff_loss_metric":
            metrics[i] = PayoffLoss_metric(input_layer, 1, 1, pure_strategies_per_player)
        elif elem == "mse_custom":
            metrics[i] = make_MSE()

    # Compile the model
    model.compile(loss=create_hydra_sse_loss(), optimizer=optimizer, metrics=metrics)

    # Return the created model
    return model



def generate_nash(game):
    '''
    Function to compute Nash equilibrium  based on the classical methods
    '''

    equilibrium = nash.Game(game[0], game[1])

    nash_support_enumeration = []
    nash_lemke_howson_enumeration = []
    nash_vertex_enumeration = []

    for eq in equilibrium.support_enumeration():
        nash_support_enumeration.append(eq)

    for eq in equilibrium.lemke_howson_enumeration():
        nash_lemke_howson_enumeration.append(eq)

    for eq in equilibrium.vertex_enumeration():
        nash_vertex_enumeration.append(eq)

    return nash_support_enumeration, nash_lemke_howson_enumeration, nash_vertex_enumeration
