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
import time
import Standard_Games

# Suppress warnings. (To mute Nashpy's verbose warnings about degenerate games)
warnings.filterwarnings("ignore")

# Global Variables
# ************************
# Output
NUMBER_OF_SAMPLES = 5000

# Game Settings
MAXIMUM_EQUILIBRIA_PER_GAME = 20
PLAYER_NUMBER = 2
PURE_STRATEGIES_PER_PLAYER = [2, 2]
# Game types: "Random" ; "Coordination" ; "Volunteer_Dilemma" ; "Prisoner_Dilemma"
GAME_TYPE = "Random"

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
def generate_game(player_number, strategies_per_player, use_gambit, game_type):
    """
    Function to generate a game.
    """

    if not use_gambit:
        game = np.zeros((player_number,) + tuple(strategies_per_player))

        for i in range(player_number):
            game[i] = np.random.rand(*tuple(strategies_per_player)).astype(np.float32)
    else:
        game = np.zeros((player_number,) + tuple(strategies_per_player), dtype=gambit.Rational)

        if game_type == "Random":
            for i in range(player_number):
                game[i] = np.random.randint(0, 1e6, tuple(strategies_per_player))

        elif game_type == "Coordination":
            game = Standard_Games.coordination_game(number_of_samples=1,
                                                    random_number_range=1e6,
                                                    number_of_players=player_number,
                                                    strategies_number=max(strategies_per_player),
                                                    data_type=gambit.Rational)[0]

        elif game_type == "Volunteer_Dilemma":
            game = Standard_Games.volunteer_dilemma(number_of_samples=1,
                                                    random_number_range=1e6,
                                                    number_of_players=player_number,
                                                    data_type=gambit.Rational)[0]

        elif game_type == "Prisoner_Dilemma":
            game = Standard_Games.prisoner_dilemma(number_of_samples=1,
                                                   random_number_range=1e6,
                                                   data_type=gambit.Rational)[0]

    return game


# ************************
def check_standard_game_strategy_number(strategies_per_player, game_type):
    """
    Function to check the game size matches the requirements of the specified game type.
    """

    if game_type == "Coordination":
        # Check if the game is symmetric
        assert strategies_per_player.count(strategies_per_player[0]) == len(strategies_per_player), \
            "The game " + str(strategies_per_player) + " is not symmetric, which is necessary for Coordination Game."
    elif game_type == "Volunteer_Dilemma":
        assert strategies_per_player.count(2) == len(strategies_per_player), \
            "The game Volunteer's Dilemma only allows for 2 strategies per player (Current game shape: " + str(strategies_per_player) + ")."
    elif game_type == "Prisoner_Dilemma":
        assert len(strategies_per_player) == 2,\
            "Prisoner's Dilemma only has two players (Current game shape: " + str(strategies_per_player) + ")."

        assert strategies_per_player.count(2) == len(strategies_per_player), \
            "The game Prisoner's Dilemma only allows for 2 strategies per player (Current game shape: " + str(strategies_per_player) + ")."


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
                     filter_pure_strategies, discard_single_equilibrium_games, use_gambit, timeout_per_sample,
                     game_type):
    """
    Function to generate games and compute their Nash equilibria
    """

    # Set random seed
    np.random.seed(int((time.time() + process_index) * 1000))

    # Check Python version
    if use_gambit:
        assert sys.version_info.major == 2, 'Gambit works only in Python 2.'
    else:
        assert sys.version_info.major == 3, 'Nashpy works only in Python 3.'

    # Check if Nashpy is not assigned for solving games with more than 2 players, as it is unable to work on those games
    if not use_gambit:
        assert player_number == 2, "Nashpy is unable to solve games with more than 2 players."
        assert game_type == "Random", "Only randomly generated games are supported in Nashpy mode."

    # Check if enough strategy numbers are given for players
    try:
        assert len(strategies_per_player) == player_number, \
            'Number of strategies for at least one player is not defined.'
    except TypeError:
        raise Exception('Number of strategies for players should be an iterable.')

    # If standard games are being generated, check number of strategies of each player based on the game rules
    check_standard_game_strategy_number(strategies_per_player, game_type)

    # Create an empty numpy array with proper shape for games and Nash equilibria
    games = np.zeros(((num_games, player_number) + tuple(strategies_per_player)), dtype=np.float32)
    nashes = np.zeros((num_games, max_nashes, player_number, max(strategies_per_player)), dtype=np.float32)

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
                              use_gambit=use_gambit,
                              game_type=game_type)

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
            if use_gambit:
                games[count] = (g.astype(np.float32) - np.min(g)) / (np.max(g) - np.min(g))
            else:
                games[count] = g

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
                            timeout_per_sample, game_type):
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
                                                     discard_single_equilibrium_games, use_gambit, timeout_per_sample,
                                                     game_type))

        processes[processCounter].start()

    processes[cpu_cores - 1] = mp.Process(target=generate_dataset, args=(
        games, equilibria, (cpu_cores - 1), numGenerated, number_of_samples - (cpu_cores - 1) * processQuota,
        max_equilibria_per_game, player_number, strategies_per_player, discard_non_mixed_strategy_games,
        filter_pure_strategies, discard_single_equilibrium_games, use_gambit, timeout_per_sample, game_type))

    processes[cpu_cores - 1].start()

    # Rung a thread to keep track of number of generated samples
    trackerThread = threading.Thread(target=track_sampleGeneration_progress, args=(numGenerated, number_of_samples))
    trackerThread.start()

    # Waiting for the threads to finish
    for processCounter in range(cpu_cores):
        processes[processCounter].join()
    trackerThread.join()

    # Stack up the numpy arrays resulted from each thread
    games = np.vstack(games)
    equilibria = np.vstack(equilibria)

    # Save the generated arrays on the local drive
    np.save("./Datasets/" + games_dataset_name, games)
    np.save("./Datasets/" + equilibria_dataset_name, equilibria)


# ************************
def construct_output_filename(pure_strategies_per_player, player_number, number_of_samples, dataset_number=1):
    # Construct the file names
    strategy_str = ''

    for strategy_element in pure_strategies_per_player[:-1]:
        strategy_str += str(strategy_element) + 'x'

    strategy_str += str(pure_strategies_per_player[-1])

    games_dataset_name = 'Games-{}_{}P-{}_{:.0E}'.format(dataset_number, player_number, strategy_str, number_of_samples)
    equilibria_dataset_name = 'Equilibria-{}_{}P-{}_{:.0E}'.format(dataset_number, player_number, strategy_str, number_of_samples)

    return games_dataset_name, equilibria_dataset_name


# ************************
# Main script
if __name__ == "__main__":
    games_dataset_name, equilibria_dataset_name = construct_output_filename(pure_strategies_per_player=PURE_STRATEGIES_PER_PLAYER,
                                                                            player_number=PLAYER_NUMBER,
                                                                            number_of_samples=NUMBER_OF_SAMPLES)

    multi_process_generator(games_dataset_name=games_dataset_name,
                            equilibria_dataset_name=equilibria_dataset_name,
                            number_of_samples=NUMBER_OF_SAMPLES,
                            max_equilibria_per_game=MAXIMUM_EQUILIBRIA_PER_GAME,
                            player_number=PLAYER_NUMBER,
                            strategies_per_player=PURE_STRATEGIES_PER_PLAYER,
                            discard_non_mixed_strategy_games=DISCARD_NON_MIXED_STRATEGY_GAMES,
                            filter_pure_strategies=FILTER_PURE_STRATEGIES,
                            discard_single_equilibrium_games=DISCARD_SINGLE_EQUILIBRIUM_GAMES,
                            use_gambit=USE_GAMBIT,
                            cpu_cores=CPU_CORES,
                            timeout_per_sample=TIMEOUT_PER_SAMPLE,
                            game_type=GAME_TYPE)
