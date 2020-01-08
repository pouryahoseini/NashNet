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

# Suppress warnings. (To mute Nashpy's verbose warnings about degenerate games)
warnings.filterwarnings("ignore")

# Global Variables
# ************************
# Output
GENERATED_GAMES_DATASET_NAME = 'Games'
GENERATED_EQUILIBRIA_DATASET_NAME = 'Equilibria'
NUMBER_OF_SAMPLES = 1000000

# Game Settings
MAXIMUM_EQUILIBRIA_PER_GAME = 20
PLAYER_NUMBER = 3
PURE_STRATEGIES_PER_PLAYER = [3, 4, 3]

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
TIMEOUT_PER_SAMPLE = 2


# ************************
def generate_game(players, strategies_per_player):
    """
	Function to generate a game with values between 0 and 1
	"""

    if not USE_GAMBIT:
        game = np.zeros((players,) + tuple(strategies_per_player))

        for i in range(players):
            game[i] = np.random.rand(*tuple(strategies_per_player))
    else:
        game = np.zeros((players,) + tuple(strategies_per_player), dtype=gambit.Rational)

        for i in range(players):
            game[i] = np.random.randint(0, 1e6, tuple(strategies_per_player))

    return game


# ************************
def compute_nash(game):
    """
	Function to compute Nash equilibrium of a game
	"""

    players = game.shape[0]
    splitGames = np.split(game, players, axis=0)

    for i, g in enumerate(splitGames):
        splitGames[i] = np.squeeze(g)

    nash_eq = []
    if not USE_GAMBIT:
        games_set = nashpy.Game(*splitGames)

        nash_eqs = games_set.support_enumeration()
        for eq in nash_eqs:
            eq_fixed_size = np.array(list(itertools.zip_longest(*eq, fillvalue=np.nan))).T
            nash_eq.append(eq_fixed_size)
        # 	for eq in games_set.lemke_howson_enumeration():
        # 		nash.append(eq)
        #
        # 	for eq in games_set.vertex_enumeration():
        # 		nash.append(eq)
    else:
        games_set = gambit.Game.from_arrays(*splitGames)

        if PLAYER_NUMBER > 2:
            solver = gambit.nash.ExternalGlobalNewtonSolver()
        else:
            solver = gambit.nash.ExternalEnumMixedSolver()

        nash_eqs = solver.solve(games_set)
        for eq in nash_eqs:
            eq_list = ast.literal_eval(str(eq._profile))
            eq_fixed_size = np.array(list(itertools.izip_longest(*eq_list, fillvalue=np.nan))).T
            nash_eq.append(eq_fixed_size.astype(np.float))

    return nash_eq


# ************************
def convertToN(array, max_nashes, players, strategies):
    '''
	Function to set the number of listed equilibria to a fixed value
	'''

    # Create numpy array to store Nash equilibria
    nash = np.zeros((max_nashes, players, strategies))

    # Create itertools cycle
    cycle = itertools.cycle(array)

    # Iterate through list and indices
    for i, elem in zip(range(max_nashes), cycle):
        nash[i] = elem

    return nash


# ************************
def discardNonMixedStrategyGames(equilibria):
    '''
	Function to discard games that contain a pure strategy equilibrium
	'''

    skip = False

    for eq in equilibria:
        unravelled_equilibrium = np.ravel(eq)
        if np.logical_or(np.logical_or(unravelled_equilibrium == 0, unravelled_equilibrium == 1), unravelled_equilibrium == np.nan).all():
            skip = True
            break

    return skip


# ************************
def filterPureStrategies(equilibria):
    '''
	Function to filter out pure strategies from set of equilibria of each game
	'''

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
    '''
	Function to discard games that have only one Nash equilibrium.
	'''

    # Check the number of equilibria
    if len(equilibria) == 1:
        return True
    else:
        return False


# ************************
def generate_dataset(output_games, output_equilibria, process_index, num_generated, num_games=NUMBER_OF_SAMPLES,
                     max_nashes=MAXIMUM_EQUILIBRIA_PER_GAME, players=PLAYER_NUMBER,
                     strategies=PURE_STRATEGIES_PER_PLAYER):
    """
	Function to generate games and compute their Nash equilibria
	"""

    # Check Python version
    if USE_GAMBIT:
        assert sys.version_info.major == 2, 'Gambit works only in Python 2.'
    else:
        assert sys.version_info.major == 3, 'Nashpy works only in Python 3.'

    # Check if Nashpy is not assigned for solving games with more than 2 players, as it is unable to work on those games
    if not USE_GAMBIT:
        assert PLAYER_NUMBER == 2, "Nashpy is unable to solve games with more than 2 players."

    # Check if enough strategy numbers are given for players
    try:
        assert len(PURE_STRATEGIES_PER_PLAYER) == PLAYER_NUMBER, \
            'Number of strategies for at least one player is not defined.'
    except TypeError:
        raise Exception('Number of strategies for players should be an iterable.')

    # Create an empty numpy array with proper shape for games and Nash equilibria
    games = np.zeros((num_games, players) + tuple(strategies))
    nashes = np.zeros((num_games, max_nashes, players, max(strategies)))

    # Loop
    count = 0

    while count < num_games:
        # Put game generation in a try statement and catch runtime warning if
        # the game is degenerate (in the case of Nashpy), retry without incrementing count

        # Set the timeout
        signal.alarm(0)
        signal.alarm(TIMEOUT_PER_SAMPLE)

        try:
            # Generate a game
            g = generate_game(players=players, strategies_per_player=strategies)

            # Get the nash equilibria
            eq = compute_nash(g)

            # Check to remove degenerate games in Gambit (where all the probabilities are zero)
            if np.sum(sum(eq)) == 0:
                continue

            # If enabled, discard games containing pure strategy equilibria
            skip = skip2 = False

            if DISCARD_NON_MIXED_STRATEGY_GAMES:
                skip = discardNonMixedStrategyGames(eq)
            elif FILTER_PURE_STRATEGIES:  # If enabled, filter out pure strategy equilibria
                eq, skip = filterPureStrategies(eq)

            # If enabled, discard games with just one Nash equilibrium
            if DISCARD_SINGLE_EQUILIBRIUM_GAMES:
                skip2 = discardSingleEquilibriumGames(eq)

            if skip or skip2:
                continue

            # If it got here, game is not degenerate
            games[count] = g
            if USE_GAMBIT:
                games[count] = (games[count].astype(np.float) - np.min(games[count])) / (
                            np.max(games[count]) - np.min(games[count]))

            # Set the number of Nash equilibria to a predefined one
            nash_nSet = convertToN(eq, max_nashes=max_nashes, players=players, strategies=max(strategies))

            # Store nash in nashes
            nashes[count] = nash_nSet

            # Increment Count
            count = count + 1
            num_generated[process_index] = count

        # In case of Nashpy a degenerate game produces this warning
        # In general, a timeout also produces this warning
        except RuntimeWarning:
            if USE_GAMBIT:
                if players > 2:
                    os.system('killall gambit-gnm >/dev/null 2>&1')
                else:
                    os.system('killall gambit-enummixed >/dev/null 2>&1')
            continue

    # Modify the list of games and equilibria
    output_games[process_index] = games
    output_equilibria[process_index] = nashes


# ************************
def timeout_handler(signum, frame):
    """
    Function to handle timeout in generating a sample
    """

    raise RuntimeWarning("Timeout")


# ************************
def track_sampleGeneration_progress(num_generated, num_samples):
    '''
	Function to print the progress of the sample generation
	'''

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
# The main script
games = mp.Manager().list([None] * CPU_CORES)
equilibria = mp.Manager().list([None] * CPU_CORES)
processes = [None] * CPU_CORES
numGenerated = mp.Array('i', [0] * (CPU_CORES))
processQuota = int(math.ceil(float(NUMBER_OF_SAMPLES) / CPU_CORES))

# Set the timeout handling in generating each sample
signal.signal(signal.SIGALRM, timeout_handler)

# Running the threads to generate the dataset
for processCounter in range(CPU_CORES - 1):
    processes[processCounter] = mp.Process(target=generate_dataset,
                                           args=(games, equilibria, processCounter, numGenerated, processQuota))
    processes[processCounter].start()
processes[CPU_CORES - 1] = mp.Process(target=generate_dataset, args=(
    games, equilibria, (CPU_CORES - 1), numGenerated, NUMBER_OF_SAMPLES - (CPU_CORES - 1) * processQuota))
processes[CPU_CORES - 1].start()

# Rung a thread to keep track of number of generated samples
trackerThread = threading.Thread(target=track_sampleGeneration_progress, args=(numGenerated, NUMBER_OF_SAMPLES))
trackerThread.start()

# Waiting for the threads to finish
for processCounter in range(CPU_CORES):
    processes[processCounter].join()
trackerThread.join()

# Stack up the numpy arrays resulted from each thread
games = np.vstack(games[:])
equilibria = np.vstack(equilibria[:])

np.save("./Datasets/" + GENERATED_GAMES_DATASET_NAME, games)
np.save("./Datasets/" + GENERATED_EQUILIBRIA_DATASET_NAME, equilibria)
