# Headers
import numpy as np
import itertools
import multiprocessing as mp
import threading
import math
import time
import warnings

# Suppress warnings. (To mute Nashpy's verbose warnings about degenerate games)
warnings.filterwarnings("ignore")


# Global Variables
# ************************
# Output
GENERATED_GAMES_DATASET_NAME = 'Games'
GENERATED_EQUILIBRIA_DATASET_NAME = 'Equilibria'
NUMBER_OF_SAMPLES = 100

# Game Settings
MAXIMUM_EQUILIBRIA_PER_GAME = 25
PLAYER_NUMBER = 3
PURE_STRATEGIES_PER_PLAYER = 3

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


# ************************
def generate_game(players=PLAYER_NUMBER, size=PURE_STRATEGIES_PER_PLAYER):
    """
	Function to generate a game with values between 0 and 1
	"""

    if not USE_GAMBIT:
        game = np.zeros((players,) + (size,) * players)

        for i in range(players):
            game[i] = np.random.rand(*((size,) * players))
    else:
        game = np.zeros((players,) + (size,) * players, dtype=gambit.Rational)

        for i in range(players):
            game[i] = np.random.randint(0, 1e6, (size,) * players)

    return game


# ************************
def compute_nash(game, players=PLAYER_NUMBER):
    """
	Function to compute Nash equilibrium of a game
	"""

    splitGames = np.split(game, players, axis=0)

    for i, g in enumerate(splitGames):
        splitGames[i] = np.squeeze(g)

    nash_eq = []
    if not USE_GAMBIT:
        games_set = nashpy.Game(*splitGames)

        nash_eqs = games_set.support_enumeration()
        for eq in nash_eqs:
            nash_eq.append(eq)
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
            nash_eq.append(np.reshape(np.array(eq._profile).astype(np.float), (players, -1)))

    return nash_eq


# ************************
def convertToN(array, max_nashes=MAXIMUM_EQUILIBRIA_PER_GAME, players=PLAYER_NUMBER,
               strategies=PURE_STRATEGIES_PER_PLAYER):
    '''
	Function to set the number of listed equilibria to a fixed value
	'''

    # Create numpy array to store Nash equilibria
    nash = np.zeros((max_nashes, players, strategies))

    # Create itertools cycle
    cycle = itertools.cycle(array)

    # Iterate through list and indices
    for i, elem in zip(range(max_nashes), cycle):
        nash[i] = np.array(elem)

    return nash


# ************************
def discardNonMixedStrategyGames(equilibria):
    '''
	Function to discard games that contain a pure strategy equilibrium
	'''

    skip = False

    for eq in equilibria:
        ravelledEquilibrium = np.ravel(eq)
        if np.logical_or(ravelledEquilibrium == 0, ravelledEquilibrium == 1).all():
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
        ravelledEquilibrium = np.ravel(eq)
        if np.logical_and(ravelledEquilibrium > 0, ravelledEquilibrium < 1).any():
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
    '''
	Function to generate games and compute their Nash equilibria
	'''

    # Create an empty numpy array with proper shape for games and nashes
    games = np.zeros((num_games, players) + (strategies, ) * players)
    nashes = np.zeros((num_games, max_nashes, players, strategies))

    # Loop
    count = 0

    while count < num_games:
        # Put game generation in a try statement and catch runtime warning if
        # the game is degenerate (in the case of Nashpy), retry without incrementing count
        try:
            # Generate a game
            g = generate_game(players=players, size=strategies)

            # Get the nash equilibria
            eq = compute_nash(g)

            # Check to remove degenerate games in Gambit (where all the probabilities are zero)
            sum_prob = np.sum(sum(eq))
            if sum_prob == 0:
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
                games[count] = (games[count].astype(np.float) - np.min(games[count])) / (np.max(games[count]) - np.min(games[count]))

            # Set the number of Nash equilibria to a predefined one
            nash_nSet = convertToN(eq, max_nashes=max_nashes, players=players, strategies=strategies)

            # Store nash in nashes
            nashes[count] = nash_nSet

            # Increment Count
            count = count + 1
            num_generated[process_index] = count

        except RuntimeWarning:
            continue
    # 			print("Warning - game is degenerate")

    # Modify the list of games and equilibria
    output_games[process_index] = games
    output_equilibria[process_index] = nashes


# ************************
def track_sampleGeneration_progress(num_generated, num_samples):
    '''
	Function to print the progress of the sample generation
	'''

    start = time.time()
    total = lastTotal = 0

    while (total < num_samples):
        total = 0
        for thread_genNum in num_generated:
            total += thread_genNum

        if ((total - lastTotal) >= 250):
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
