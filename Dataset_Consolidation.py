import shutil
import os
import numpy as np

RELATIVE_SPLITTING_ADDRESS = "2P/4x4/"
ROOT_ADDRESS = "/home/pourya/Documents/NashNet/Datasets/"
SPLIT_FILES_FOLDER = "/Split_Files/"
ORIGINAL_FILES_FOLDER = "Individual_Games/"
UNSORTED_FILES_FOLDER = "Unsorted/"
SPLIT_FILE_SAMPLES = 100000


# These are scripts to make organizing the generated data less shitty
# Simple utility func
def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Moves games from unsorted to their proper folder under Individual_Games
def move_data_to_folder(dataset_path, delete_duplicates=True):
    unsorted_path = dataset_path + UNSORTED_FILES_FOLDER

    print('Moving dataset files')
    # Exclude directories
    files = os.listdir(unsorted_path)

    # Find if
    for f in files:
        # Get the number of players
        x = f[f.find("_") + 1:]
        x = x[:x.find("P")]
        num_players = int(x)

        players_folder = str(num_players) + "P/"

        # Check if nP Directory exists. If not, create it
        ensure_path_exists(dataset_path + players_folder)

        # Get the strategies
        x = f[f.find("_") + 1:]
        x = x[x.find("-") + 1:]

        strategies_folder = x[:x.find("_")] + "/"
        individual_games_folder = dataset_path + players_folder + strategies_folder + ORIGINAL_FILES_FOLDER

        # Check if strategies folder exists. If not, create it
        # Also ensure that the individual_games subfolder exists
        ensure_path_exists(dataset_path + players_folder + strategies_folder)
        ensure_path_exists(individual_games_folder)

        # Check if f is currently in individual_games folder
        if f in os.listdir(individual_games_folder):
            print("Duplicate game found: ", f)
            if delete_duplicates:
                print("\t Deleting...")
                os.remove(unsorted_path + f)
            else:
                print("\t Ignoring...")
        # Otherwise, move it into the folder
        else:
            shutil.move(unsorted_path + f, individual_games_folder + f)


# Checks to see if any new games have been added, and creates combined Equilibria and Games files
#   Input - Folder, which contains a subfolder named Individual Games
def combine_games(path, force_regen=False):
    # Get all files in the directory
    files = os.listdir(path)

    # Check and see if a combined games file and tracker exist yet
    if "Combined_Games.npy" in files and \
            "Combined_Equilibria.npy" in files and \
            "Combined_List.pkl" in files:
        # They exist - Check and see if there were any new files added
        gen = True  # I'm lazy - for now it's just always gonna generate.

    # If gen, then create combined games and equilibria files
    # Get all files


def split_games(path, size=SPLIT_FILE_SAMPLES):
    print('Splitting the dataset files')
    # Get files and sort them
    full_path = path + ORIGINAL_FILES_FOLDER
    dest_path = path + SPLIT_FILES_FOLDER
    files = os.listdir(full_path)
    equilibria = [x for x in files if "Equilibria" in x]
    games = [x for x in files if "Games" in x]

    # Check to make sure that there an equal amount of games and equilibria
    assert len(games) == len(equilibria)

    # Sort them, then start iterating over them. Break each into sets of *size* games
    equilibria.sort()
    games.sort()

    # Prime the loop. Open initial game and equilibria, then create temp writing np array based on their shapes
    g_init = np.load(full_path + games[0])
    e_init = np.load(full_path + equilibria[0])
    g_acc = np.zeros((size,) + g_init.shape[1:])
    e_acc = np.zeros((size,) + e_init.shape[1:])
    acc_ctr = 0
    if g_init.shape[0] < size or g_init.shape[0] % size != 0:
        raise ValueError("Shape is smaller than size, or size does not evenly divide shape.")

    for i in range(len(games)):
        # Open the npy
        g = np.load(full_path + games[i])
        e = np.load(full_path + equilibria[i])

        # Iterate through arrays and write the new shit to files
        ensure_path_exists(dest_path)

        # Only use more recent generated games where shape is (## of games, <Game stuff>)
        for j in range(int(g.shape[0] / size)):
            # Copy over section into accumulator arrays
            g_acc[0:size] = g[j * size:(j + 1) * size]
            e_acc[0:size] = e[j * size:(j + 1) * size]

            # Save the accumulator arrays, and increment acc_ctr
            np.save(dest_path + "Games_" + str(acc_ctr) + ".npy", g_acc)
            np.save(dest_path + "Equilibria_" + str(acc_ctr) + ".npy", e_acc)
            acc_ctr += 1


move_data_to_folder(dataset_path=ROOT_ADDRESS)
split_games(ROOT_ADDRESS + RELATIVE_SPLITTING_ADDRESS)
