import Dataset_Generator

# Global Variables
NUMBER_OF_SAMPLES = 2500000
MAX_GAME_SIZE = 100000
DISCARD_NON_MIXED_STRATEGY_GAMES = False
FILTER_PURE_STRATEGIES = False
DISCARD_SINGLE_EQUILIBRIUM_GAMES = False
CPU_CORES = 8
USE_GAMBIT = True

DATASET_NUMBER = 5
# DATASET_NUMBER = 1
PURE_STRATEGIES_PER_PLAYER = [[2, 2], [2, 2, 2, 2, 2], [10, 10], [2, 3], [10, 8], [2, 3, 4], [3, 3], [4, 4], [2, 2, 2], [2, 2, 2, 2]]
PLAYER_NUMBER = [len(x) for x in PURE_STRATEGIES_PER_PLAYER]
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
    GAMES_DATASET_NAME.append('Games-{}_{}P-{}_{:.0E}'.format(DATASET_NUMBER, PLAYER_NUMBER[i], strategy_str, MAX_GAME_SIZE))
    EQUILIBRIA_DATASET_NAME.append('Equilibria-{}_{}P-{}_{:.0E}'.format(DATASET_NUMBER, PLAYER_NUMBER[i], strategy_str, MAX_GAME_SIZE))

# Check if the lists match
assert NUMBER_OF_SAMPLES % MAX_GAME_SIZE == 0,\
    "The number of samples must be evenly divisible by max game size."

# Generate the datasets
for i in range(len(PLAYER_NUMBER)):
    for j in range(int(NUMBER_OF_SAMPLES/MAX_GAME_SIZE)):
        print('+++Starting to generate a dataset of games with size ' + str(PURE_STRATEGIES_PER_PLAYER[i]) + ' - number: ' + str(j + 1))

        Dataset_Generator.multi_process_generator(games_dataset_name=GAMES_DATASET_NAME[i]+"_"+str(j),
                                                  equilibria_dataset_name=EQUILIBRIA_DATASET_NAME[i]+"_"+str(j),
                                                  number_of_samples=MAX_GAME_SIZE,
                                                  max_equilibria_per_game=MAXIMUM_EQUILIBRIA_PER_GAME,
                                                  player_number=PLAYER_NUMBER[i],
                                                  strategies_per_player=PURE_STRATEGIES_PER_PLAYER[i],
                                                  discard_non_mixed_strategy_games=DISCARD_NON_MIXED_STRATEGY_GAMES,
                                                  filter_pure_strategies=FILTER_PURE_STRATEGIES,
                                                  discard_single_equilibrium_games=DISCARD_SINGLE_EQUILIBRIUM_GAMES,
                                                  use_gambit=USE_GAMBIT,
                                                  cpu_cores=CPU_CORES,
                                                  timeout_per_sample=TIMEOUT_PER_SAMPLE)

    print('***Finished generating ' + str(NUMBER_OF_SAMPLES) + ' samples of size ' + str(PURE_STRATEGIES_PER_PLAYER[i]) + ' - number: ' + str(j + 1) + '\n')
