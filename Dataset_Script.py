import Dataset_Generator

# ***********************
# Global Variables
NUMBER_OF_SAMPLES = 2500000
MAX_GAME_SIZE = 100000
DISCARD_NON_MIXED_STRATEGY_GAMES = False
FILTER_PURE_STRATEGIES = False
DISCARD_SINGLE_EQUILIBRIUM_GAMES = False
CPU_CORES = 8
USE_GAMBIT = True

DATASET_NUMBER = 1
PURE_STRATEGIES_PER_PLAYER = [[2, 2], [2, 2, 2, 2, 2], [10, 10], [2, 3], [10, 8], [2, 3, 4], [3, 3], [4, 4], [2, 2, 2], [2, 2, 2, 2]]
TIMEOUT_PER_SAMPLE = 5
MAXIMUM_EQUILIBRIA_PER_GAME = 20


# ***********************
# Extract list of player numbers
player_number = [len(x) for x in PURE_STRATEGIES_PER_PLAYER]

# Construct the file names
games_dataset_name = []
equilibria_dataset_name = []
for i, strategy in enumerate(PURE_STRATEGIES_PER_PLAYER):
    strategy_str = ''
    for strategy_element in strategy[:-1]:
        strategy_str += str(strategy_element) + 'x'
    strategy_str += str(strategy[-1])
    games_dataset_name.append('Games-{}_{}P-{}_{:.0E}'.format(DATASET_NUMBER, player_number[i], strategy_str, MAX_GAME_SIZE))
    equilibria_dataset_name.append('Equilibria-{}_{}P-{}_{:.0E}'.format(DATASET_NUMBER, player_number[i], strategy_str, MAX_GAME_SIZE))

# Check if the lists match
assert NUMBER_OF_SAMPLES % MAX_GAME_SIZE == 0,\
    "The number of samples must be evenly divisible by max game size."

# Generate the datasets
for i in range(len(player_number)):
    for j in range(int(NUMBER_OF_SAMPLES / MAX_GAME_SIZE)):

        # Print a message
        print('+++Starting to generate a dataset of games with size ' + str(PURE_STRATEGIES_PER_PLAYER[i]) + ' - number: ' + str(j + 1))

        # Set the subfolder to save the data in
        save_subfolder = str(DATASET_NUMBER) + "/" + str(PURE_STRATEGIES_PER_PLAYER[i]).rstrip(")]").lstrip("[(").replace(",", "x").replace(" ", "") + "/"

        # Run the dataset generator
        Dataset_Generator.multi_process_generator(games_dataset_name=games_dataset_name[i] + "_" + str(j),
                                                  equilibria_dataset_name=equilibria_dataset_name[i] + "_" + str(j),
                                                  number_of_samples=MAX_GAME_SIZE,
                                                  max_equilibria_per_game=MAXIMUM_EQUILIBRIA_PER_GAME,
                                                  player_number=player_number[i],
                                                  strategies_per_player=PURE_STRATEGIES_PER_PLAYER[i],
                                                  discard_non_mixed_strategy_games=DISCARD_NON_MIXED_STRATEGY_GAMES,
                                                  filter_pure_strategies=FILTER_PURE_STRATEGIES,
                                                  discard_single_equilibrium_games=DISCARD_SINGLE_EQUILIBRIUM_GAMES,
                                                  use_gambit=USE_GAMBIT,
                                                  cpu_cores=CPU_CORES,
                                                  timeout_per_sample=TIMEOUT_PER_SAMPLE,
                                                  game_type="Random",
                                                  save_subfolder=save_subfolder)

    print('***Finished generating ' + str(NUMBER_OF_SAMPLES) + ' samples of size ' + str(PURE_STRATEGIES_PER_PLAYER[i]) + ' - number: ' + str(j + 1) + '\n')
