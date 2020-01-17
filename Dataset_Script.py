import Dataset_Generator

# Global Variables
NUMBER_OF_SAMPLES = 1000000
DISCARD_NON_MIXED_STRATEGY_GAMES = False
FILTER_PURE_STRATEGIES = False
DISCARD_SINGLE_EQUILIBRIUM_GAMES = False
CPU_CORES = 8
USE_GAMBIT = True

DATASET_NUMBER = 1
PLAYER_NUMBER = [2, 2, 2, 2, 3, 3, 2, 2, 2,
                 4, 5, 3, 3, 4, 3, 5, 2, 4]
PURE_STRATEGIES_PER_PLAYER = [[2, 2], [16, 16], [2, 3], [10, 20], [3, 3, 3], [4, 5, 2], [5, 5], [7, 7], [11, 11],
                              [2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2], [4, 4, 4], [3, 3, 3, 3], [5, 5, 5], [3, 3, 3, 3, 3], [20, 20], [4, 4, 4, 4]]
TIMEOUT_PER_SAMPLE = 5
MAXIMUM_EQUILIBRIA_PER_GAME = 20

# Construct the file names
GAMES_DATASET_NAME = []
EQUILIBRIA_DATASET_NAME = []
for i, strategy in enumerate(PURE_STRATEGIES_PER_PLAYER):
    strategy_str = ''
    for strategy_element in strategy[:-1]:
        strategy_str += str(strategy_element) + 'x'
    strategy_str += str(strategy[-1])
    GAMES_DATASET_NAME.append('Games-{}_{}P-{}_{:.0E}'.format(DATASET_NUMBER, PLAYER_NUMBER[i], strategy_str, NUMBER_OF_SAMPLES))
    EQUILIBRIA_DATASET_NAME.append('Equilibria-{}_{}P-{}_{:.0E}'.format(DATASET_NUMBER, PLAYER_NUMBER[i], strategy_str, NUMBER_OF_SAMPLES))

# Check if the lists match
assert len(PLAYER_NUMBER) == len(PURE_STRATEGIES_PER_PLAYER),\
    "The list of player numbers and the list of strategies have different number of elements."

# Generate the datasets
for i in range(len(PLAYER_NUMBER)):
    print('+++Starting to generate a dataset of games with size ' + str(PURE_STRATEGIES_PER_PLAYER[i]))

    Dataset_Generator.multi_process_generator(games_dataset_name=GAMES_DATASET_NAME[i],
                                              equilibria_dataset_name=EQUILIBRIA_DATASET_NAME[i],
                                              number_of_samples=NUMBER_OF_SAMPLES,
                                              max_equilibria_per_game=MAXIMUM_EQUILIBRIA_PER_GAME,
                                              player_number=PLAYER_NUMBER[i],
                                              strategies_per_player=PURE_STRATEGIES_PER_PLAYER[i],
                                              discard_non_mixed_strategy_games=DISCARD_NON_MIXED_STRATEGY_GAMES,
                                              filter_pure_strategies=FILTER_PURE_STRATEGIES,
                                              discard_single_equilibrium_games=DISCARD_SINGLE_EQUILIBRIUM_GAMES,
                                              use_gambit=USE_GAMBIT,
                                              cpu_cores=CPU_CORES,
                                              timeout_per_sample=TIMEOUT_PER_SAMPLE)

    print('***Finished generating ' + str(NUMBER_OF_SAMPLES) + ' samples of size ' + str(PURE_STRATEGIES_PER_PLAYER[i]) + '\n')
