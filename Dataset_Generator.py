# Headers
import numpy as np
import itertools
import multiprocessing as mp
import threading
import math
import time
import warnings
import ast
import sys
import signal
import os
import gc

# Suppress warnings. (To mute Nashpy's verbose warnings about degenerate games)
warnings.filterwarnings("ignore")

# Global Variables
# ************************
# Output
GAMES_DATASET_NAME = 'Games'
EQUILIBRIA_DATASET_NAME = 'Equilibria'
NUMBER_OF_SAMPLES = 300

# Game Settings
MAXIMUM_EQUILIBRIA_PER_GAME = 20
PLAYER_NUMBER = 2
PURE_STRATEGIES_PER_PLAYER = [3, 3]

# Equilibrium filtering
DISCARD_NON_MIXED_STRATEGY_GAMES = False
FILTER_PURE_STRATEGIES = False
DISCARD_SINGLE_EQUILIBRIUM_GAMES = False

# Gambit library
USE_GAMBIT = True

if USE_GAMBIT:
    import gambit  # Python 2 only
else:
    import nashpy  # Python 3 only

# Multi-threading
CPU_CORES = 8

# Timeout Setting (seconds)
TIMEOUT_PER_SAMPLE = 10


# ************************
def generate_game(player_number, strategies_per_player, use_gambit):
    """
    Function to generate a game with values between 0 and 1
    """

    if not use_gambit:
        game = np.zeros((player_number,) + tuple(strategies_per_player))

        for i in range(player_number):
            game[i] = np.random.rand(*tuple(strategies_per_player)).astype(np.float32)
    else:
        game = np.zeros((player_number,) + tuple(strategies_per_player), dtype=gambit.Rational)

        for i in range(player_number):
            game[i] = np.random.randint(0, 1e6, tuple(strategies_per_player))

    return game


# ************************
def compute_nash(game, use_gambit):
    """
    Function to compute Nash equilibrium of a game
    """

    player_number = game.shape[0]
    splitGames = np.split(game, player_number, axis=0)

    for i, g in enumerate(splitGames):
        splitGames[i] = np.squeeze(g)

    nash_eq = []
    if not use_gambit:
        games_set = nashpy.Game(*splitGames)

        nash_eqs = games_set.support_enumeration()
        for eq in nash_eqs:
            eq_fixed_size = np.array(list(itertools.zip_longest(*eq, fillvalue=np.nan))).T
            nash_eq.append(eq_fixed_size.astype(np.float32))
        # 	for eq in games_set.lemke_howson_enumeration():
        # 		nash.append(eq)
        #
        # 	for eq in games_set.vertex_enumeration():
        # 		nash.append(eq)
    else:
        games_set = gambit.Game.from_arrays(*splitGames)

        if player_number > 2:
            solver = gambit.nash.ExternalGlobalNewtonSolver()
        else:
            solver = gambit.nash.ExternalGlobalNewtonSolver()
            # solver = gambit.nash.ExternalEnumMixedSolver()

        nash_eqs = solver.solve(games_set)
        for eq in nash_eqs:
            eq_list = ast.literal_eval(str(eq._profile))
            eq_fixed_size = np.array(list(itertools.izip_longest(*eq_list, fillvalue=np.nan))).T
            nash_eq.append(eq_fixed_size.astype(np.float32))

    return nash_eq


# ************************
def convertToN(array, max_nashes, player_number, strategies_per_player):
    """
    Function to set the number of listed equilibria to a fixed value
    """

    # Create numpy array to store Nash equilibria
    nash = np.zeros((max_nashes, player_number, strategies_per_player))

    # Create itertools cycle
    cycle = itertools.cycle(array)

    # Iterate through list and indices
    for i, elem in zip(range(max_nashes), cycle):
        nash[i] = elem

    return nash


# ************************
def discardNonMixedStrategyGames(equilibria):
    """
    Function to discard games that contain a pure strategy equilibrium
    """

    skip = False

    for eq in equilibria:
        unravelled_equilibrium = np.ravel(eq)
        if np.logical_or(np.logical_or(unravelled_equilibrium == 0, unravelled_equilibrium == 1), unravelled_equilibrium == np.nan).all():
            skip = True
            break

    return skip


# ************************
def filterPureStrategies(equilibria):
    """
    Function to filter out pure strategies from set of equilibria of each game
    """

    skip = True
    filteredEquilibria = []

    for eq in equilibria:
        unravelled_equilibrium = np.ravel(eq)
        if np.logical_and(unravelled_equilibrium > 0, unravelled_equilibrium < 1).any():
            skip = False
            filteredEquilibria += (eq,)

    return filteredEquilibria, skip


# ************************
def discardSingleEquilibriumGames(equilibria):
    """
    Function to discard games that have only one Nash equilibrium.
    """

    # Check the number of equilibria
    if len(equilibria) == 1:
        return True
    else:
        return False


# ************************
def generate_dataset(output_games, output_equilibria, process_index, num_generated, num_games,
                     max_nashes, player_number, strategies_per_player, discard_non_mixed_strategy_games,
                     filter_pure_strategies, discard_single_equilibrium_games, use_gambit, timeout_per_sample):
    """
    Function to generate games and compute their Nash equilibria
    """

    # Check Python version
    if use_gambit:
        assert sys.version_info.major == 2, 'Gambit works only in Python 2.'
    else:
        assert sys.version_info.major == 3, 'Nashpy works only in Python 3.'

    # Check if Nashpy is not assigned for solving games with more than 2 players, as it is unable to work on those games
    if not use_gambit:
        assert player_number == 2, "Nashpy is unable to solve games with more than 2 players."

    # Check if enough strategy numbers are given for players
    try:
        assert len(strategies_per_player) == player_number, \
            'Number of strategies for at least one player is not defined.'
    except TypeError:
        raise Exception('Number of strategies for players should be an iterable.')

    # Create an empty numpy array with proper shape for games and Nash equilibria
    temp_games_filename = os.path.join("./Datasets/", "temp_games_process{}.dat".format(process_index))
    temp_eq_filename = os.path.join("./Datasets/", "temp_equilibria_process{}.dat".format(process_index))
    games = np.memmap(temp_games_filename, shape=((num_games, player_number) + tuple(strategies_per_player)), dtype=np.float32, mode='w+')
    nashes = np.memmap(temp_eq_filename, shape=(num_games, max_nashes, player_number, max(strategies_per_player)), dtype=np.float32, mode='w+')

    # Loop
    count = 0

    while count < num_games:
        # Put game generation in a try statement and catch runtime warning if
        # the game is degenerate (in the case of Nashpy), retry without incrementing count

        # Set the timeout
        signal.alarm(0)
        signal.alarm(timeout_per_sample)

        try:
            # Generate a game
            g = generate_game(player_number=player_number,
                              strategies_per_player=strategies_per_player,
                              use_gambit=use_gambit)

            # Get the nash equilibria
            eq = compute_nash(game=g, use_gambit=use_gambit)

            # Check to remove degenerate games in Gambit (where all the probabilities are zero)
            if np.sum(sum(eq)) == 0:
                continue

            # If enabled, discard games containing pure strategy equilibria
            skip = skip2 = False

            if discard_non_mixed_strategy_games:
                skip = discardNonMixedStrategyGames(eq)
            elif filter_pure_strategies:  # If enabled, filter out pure strategy equilibria
                eq, skip = filterPureStrategies(eq)

            # If enabled, discard games with just one Nash equilibrium
            if discard_single_equilibrium_games:
                skip2 = discardSingleEquilibriumGames(eq)

            if skip or skip2:
                continue

            # If it got here, game is not degenerate
            games[count] = g
            if use_gambit:
                games[count] = (games[count].astype(np.float32) - np.min(games[count])) / (
                            np.max(games[count]) - np.min(games[count]))

            # Set the number of Nash equilibria to a predefined one
            nash_nSet = convertToN(eq, max_nashes=max_nashes, player_number=player_number, strategies_per_player=max(strategies_per_player))

            # Store nash in nashes
            nashes[count] = nash_nSet

            # Increment Count
            count = count + 1
            num_generated[process_index] = count

        # In case of Nashpy a degenerate game produces this warning
        # In general, a timeout also produces this warning
        except RuntimeWarning:
            if use_gambit:
                if player_number > 2:
                    os.system('killall gambit-gnm >/dev/null 2>&1')
                else:
                    os.system('killall gambit-enummixed >/dev/null 2>&1')
            continue

    # Modify the list of games and equilibria
    output_games[process_index] = games
    output_equilibria[process_index] = nashes

    # Delete the memory mapped arrays
    del games
    del nashes
    gc.collect()
    os.remove(temp_games_filename)
    os.remove(temp_eq_filename)


# ************************
def timeout_handler(signum, frame):
    """
    Function to handle timeout in generating a sample
    """

    raise RuntimeWarning("Timeout")


# ************************
def track_sampleGeneration_progress(num_generated, num_samples):
    """
    Function to print the progress of the sample generation
    """

    start = time.time()
    total = lastTotal = 0

    while total < num_samples:
        total = 0
        for thread_genNum in num_generated:
            total += thread_genNum

        if (total - lastTotal) >= 250:
            print("Generated {} games in {} seconds".format(total, int(time.time() - start)))
            lastTotal = total

        time.sleep(0.1)

    print("Generated {} games in {} seconds".format(total, int(time.time() - start)))

    return


# ************************
def multi_process_generator(games_dataset_name, equilibria_dataset_name, number_of_samples, max_equilibria_per_game,
                            player_number, strategies_per_player, discard_non_mixed_strategy_games,
                            filter_pure_strategies, discard_single_equilibrium_games, use_gambit, cpu_cores,
                            timeout_per_sample):
    """
    Function to generate dataset on multiple processes
    """
    games = mp.Manager().list([None] * cpu_cores)
    equilibria = mp.Manager().list([None] * cpu_cores)
    processes = [None] * cpu_cores
    numGenerated = mp.Array('i', [0] * (cpu_cores))
    processQuota = int(math.floor(float(number_of_samples) / cpu_cores))

    # Set the timeout handling in generating each sample
    signal.signal(signal.SIGALRM, timeout_handler)

    # Running the threads to generate the dataset
    for processCounter in range(cpu_cores - 1):
        processes[processCounter] = mp.Process(target=generate_dataset,
                                               args=(games, equilibria, processCounter, numGenerated, processQuota,
                                                     max_equilibria_per_game, player_number, strategies_per_player,
                                                     discard_non_mixed_strategy_games, filter_pure_strategies,
                                                     discard_single_equilibrium_games, use_gambit, timeout_per_sample))

        processes[processCounter].start()

    processes[cpu_cores - 1] = mp.Process(target=generate_dataset, args=(
        games, equilibria, (cpu_cores - 1), numGenerated, number_of_samples - (cpu_cores - 1) * processQuota,
        max_equilibria_per_game, player_number, strategies_per_player, discard_non_mixed_strategy_games,
        filter_pure_strategies, discard_single_equilibrium_games, use_gambit, timeout_per_sample))

    processes[cpu_cores - 1].start()

    # Rung a thread to keep track of number of generated samples
    trackerThread = threading.Thread(target=track_sampleGeneration_progress, args=(numGenerated, number_of_samples))
    trackerThread.start()

    # Waiting for the threads to finish
    for processCounter in range(cpu_cores):
        processes[processCounter].join()
    trackerThread.join()

    # Stack up the numpy arrays resulted from each thread
    temp_agg_games_filename = os.path.join("./Datasets/", "temp_games_agg.dat")
    temp_agg_eq_filename = os.path.join("./Datasets/", "temp_equilibria_agg.dat")
    games_agg = np.memmap(temp_agg_games_filename, shape=((number_of_samples,) + games[0].shape[1:]), dtype=np.float32, mode='w+')
    equilibria_agg = np.memmap(temp_agg_eq_filename, shape=((number_of_samples,) + equilibria[0].shape[1:]), dtype=np.float32, mode='w+')

    start_index = 0
    for processCounter in range(cpu_cores):
        n_samples = games[processCounter].shape[0]
        games_agg[start_index: start_index + n_samples] = games[0]
        equilibria_agg[start_index: start_index + n_samples] = equilibria[0]
        start_index += n_samples

    # Save the generated arrays on the local drive
    np.save("./Datasets/" + games_dataset_name, games)
    np.save("./Datasets/" + equilibria_dataset_name, equilibria)

    # Delete the memory mapped arrays
    del games_agg
    del equilibria_agg
    gc.collect()
    os.remove(temp_agg_games_filename)
    os.remove(temp_agg_eq_filename)


# ************************
# Main script
if __name__ == "__main__":
    multi_process_generator(games_dataset_name=GAMES_DATASET_NAME,
                            equilibria_dataset_name=EQUILIBRIA_DATASET_NAME,
                            number_of_samples=NUMBER_OF_SAMPLES,
                            max_equilibria_per_game=MAXIMUM_EQUILIBRIA_PER_GAME,
                            player_number=PLAYER_NUMBER,
                            strategies_per_player=PURE_STRATEGIES_PER_PLAYER,
                            discard_non_mixed_strategy_games=DISCARD_NON_MIXED_STRATEGY_GAMES,
                            filter_pure_strategies=FILTER_PURE_STRATEGIES,
                            discard_single_equilibrium_games=DISCARD_SINGLE_EQUILIBRIUM_GAMES,
                            use_gambit=USE_GAMBIT,
                            cpu_cores=CPU_CORES,
                            timeout_per_sample=TIMEOUT_PER_SAMPLE)
