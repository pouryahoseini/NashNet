#Headers
import random
import numpy as np
import pandas as pd
import nashpy as nash
import itertools, time
import multiprocessing as mp
import threading
import math
import time
import warnings
warnings.filterwarnings("error")

#Global Variables
#************************
#Output
GENERATED_GAMES_DATASET_NAME = 'Games'
GENERATED_EQUILIBRIA_DATASET_NAME = 'Equilibria'
NUMBER_OF_SAMPLES = 100000

#Game Settings
MAXIMUM_EQUILIBRIA_PER_GAME = 10
PLAYER_NUMBER = 2
PURE_STRATEGIES_PER_PLAYER = 3

#Equilibrium filtering
DISCARD_NON_MIXED_STRATEGY_GAMES = True
FILTER_PURE_STRATEGIES = False
DISCARD_SINGLE_EQUILIBRIUM_GAMES = True

#Multithreading
CPU_CORES = 8


#************************
def generate_game(players = PLAYER_NUMBER, size = (PURE_STRATEGIES_PER_PLAYER, PURE_STRATEGIES_PER_PLAYER)):
	'''
	Function to generate a game with values between 0 and 1
	'''
	
	game = np.zeros((players, size[0], size[1]))
	
	for i in range(players):
		game[i] = np.random.rand(size[0], size[1])
		
	return game

#************************
def compute_nash(game, players = PLAYER_NUMBER):
	'''
	Function to compute Nash equilibrium of a game
	'''
	
	splitGames = np.split(game, players, axis = 0)

	for i, g in enumerate(splitGames):
		splitGames[i] = np.squeeze(g)
	
	nashy = nash.Game(* splitGames)
	# nash_support_enumeration = []
	# nash_lemke_howson_enumeration = []
	# nash_vertex_enumeration = []
	nash1 = []
	nash2 = []
	nash3 = []
	
	for eq in nashy.support_enumeration():
		nash1.append(eq)
		
	for eq in nashy.lemke_howson_enumeration():
		nash2.append(eq)
		
	for eq in nashy.vertex_enumeration():
		nash3.append(eq)
		
	return nash1, nash2, nash3

#************************
def convertToN(array, max_nashes = MAXIMUM_EQUILIBRIA_PER_GAME, players = PLAYER_NUMBER, strategies = PURE_STRATEGIES_PER_PLAYER):
	'''
	Function to set the number of listed equilibria to a fixed value
	'''
	
	#Create numpy array to store nashes
	nash = np.zeros((max_nashes, players, strategies))
    
	#Create itertools cycle
	cycle = itertools.cycle(array)

	#Iterate through list and indices
	for i, elem in zip(range(max_nashes), cycle):
		nash[i] = np.array(elem)

	return nash

#************************
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

#************************
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

#************************
def discardSingleEquilibriumGames(equilibria):
	'''
	Function to discard games that have only one Nash equilibrium.
	'''
	
	#Check the number of equilibria
	if len(equilibria) == 1:
		return True
	else:
		return False

#************************
def generate_dataset(output_games, output_equilibria, process_index, num_generated, num_games = NUMBER_OF_SAMPLES, max_nashes = MAXIMUM_EQUILIBRIA_PER_GAME, players = PLAYER_NUMBER, strategies = PURE_STRATEGIES_PER_PLAYER):
	'''
	Function to generate games and compute their Nash equilibria
	'''
	
	# Create an empty numpy array with proper shape for games and nashes
	games = np.zeros((num_games, players, strategies, strategies))
	nashes = np.zeros((num_games, max_nashes, players, strategies))

	# Loop
	count = 0
	
	while(count < num_games):			
		#Put game generation in a try statement and catch runtime warning if
		#	the game is degenerate, retry without incrementing count
		try:
			# Generate a game
			g = generate_game(players = players, size = (strategies, strategies))

			#Get the nash equilibria
			eq, _, _ = compute_nash(g)
			
			#If enabled, discard games containing pure strategy equilibria
			skip = skip2 = False
			
			if DISCARD_NON_MIXED_STRATEGY_GAMES:
				skip = discardNonMixedStrategyGames(eq)
			elif FILTER_PURE_STRATEGIES: #If enabled, filter out pure strategy equilibria
				eq, skip = filterPureStrategies(eq)
			
			#If enabled, discard games with just one Nash equilibrium
			if DISCARD_SINGLE_EQUILIBRIUM_GAMES:
				skip2 = discardSingleEquilibriumGames(eq)
			
			if skip or skip2:
				continue
			
			#If it got here, game is not degenerate
			games[count] = g

			#Set the number of Nash equilibria to a predefined one
			nash_nSet = convertToN(eq, max_nashes = max_nashes, players = players, strategies = strategies)

			#Store nash in nashes
			nashes[count] = nash_nSet

			#Increment Count
			count = count + 1
			num_generated[process_index] = count
	
		except RuntimeWarning:
			continue
# 			print("Warning - game is degenerate")

	#Modify the list of games and equilibria
	output_games[process_index] = games
	output_equilibria[process_index] = nashes
	
# d = {"Game":g,"Nash1-SuppEnum":sup_enum,"Nash2-LH":lemke_howson, "Nash3-VertEnum":vertex_enum}
# df = pd.DataFrame(d)

# df.to_csv("million-val.csv")

#************************
def track_sampleGeneration_progress(num_generated, num_samples):
	'''
	Function to print the progress of the sample generation
	'''
	
	start = time.time()
	total = lastTotal = 0
	
	while(total < num_samples):
		total = 0
		for thread_genNum in num_generated:
			total += thread_genNum
			
		if ((total - lastTotal) >= 250):
			print("Generated {} games in {} seconds".format(total, int(time.time() - start)))
			lastTotal = total
			
		time.sleep(0.1)
			
	print("Generated {} games in {} seconds".format(total, int(time.time() - start)))

	return

#************************
#The main script
games = mp.Manager().list([None] * CPU_CORES)
equilibria = mp.Manager().list([None] * CPU_CORES)
processes = [None] * CPU_CORES
numGenerated = mp.Array('i', [0] * (CPU_CORES))
processQuota = int(math.ceil(float(NUMBER_OF_SAMPLES) / CPU_CORES))

#Running the threads to generate the dataset
for processCounter in range(CPU_CORES - 1):
	processes[processCounter] = mp.Process(target = generate_dataset, args = (games, equilibria, processCounter, numGenerated, processQuota))
	processes[processCounter].start()
processes[CPU_CORES - 1] = mp.Process(target = generate_dataset, args = (games, equilibria, (CPU_CORES - 1), numGenerated, NUMBER_OF_SAMPLES - (CPU_CORES - 1) * processQuota))
processes[CPU_CORES - 1].start()

#Rung a thread to keep track of number of generated samples
trackerThread = threading.Thread(target = track_sampleGeneration_progress, args = (numGenerated, NUMBER_OF_SAMPLES))
trackerThread.start()

#Waiting for the threads to finish
for processCounter in range(CPU_CORES):
	processes[processCounter].join()
trackerThread.join()

#Stack up the numpy arrays resulted from each thread
games = np.vstack(games[: ])
equilibria = np.vstack(equilibria[: ])

np.save("./Datasets/" + GENERATED_GAMES_DATASET_NAME, games)
np.save("./Datasets/" + GENERATED_EQUILIBRIA_DATASET_NAME, equilibria)

