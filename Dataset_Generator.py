#Headers
import random
import numpy as np
import pandas as pd
import nashpy as nash
import itertools, time
import warnings
warnings.filterwarnings("error")

#Global Variables
GENERATED_GAMES_DATASET_NAME = 'Games'
GENERATED_EQUILIBRIA_DATASET_NAME = 'Equilibria'
NUMBER_OF_SAMPLES = 1000000

MAXIMUM_EQUILIBRIA_PER_GAME = 10
PLAYER_NUMBER = 2
PURE_STRATEGIES_PER_PLAYER = 3

DISCARD_NON_MIXED_STRATEGY_GAMES = False
FILTER_PURE_STRATEGIES = False


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
def compute_nash(game):
	'''
	Function to compute Nash equilibrium of a game
	'''
	
	nashy = nash.Game(game[0],game[1])
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
def generate_dataset(num_games = NUMBER_OF_SAMPLES, max_nashes = MAXIMUM_EQUILIBRIA_PER_GAME, players = PLAYER_NUMBER, strategies = PURE_STRATEGIES_PER_PLAYER):
	'''
	Function to generate games and compute their Nash equilibria
	'''
	
	# Create an empty numpy array with proper shape for games and nashes
	games = np.zeros((num_games, players, strategies, strategies))
	nashes = np.zeros((num_games, max_nashes, players, strategies))

	# Loop
	start = time.time()
	count = 0
	num_degen = 0
	printed = False
	
	while(count < num_games):
		if (count % 250 == 0) and not printed:
			print("Generated {} games in {} seconds with {} degenerate games.".format(count, time.time() - start, num_degen))
			printed = True
			
		#Put game generation in a try statement and catch runtime warning if
		#	the game is degenerate, retry without incrementing count
		try:
			# Generate a game
			g = generate_game(players = players, size = (strategies, strategies))

			#Get the nash equilibria
			eq, _, _ = compute_nash(g)
			
			#If enabled, discard games containing pure strategy equilibria
			skip = False
			
			if DISCARD_NON_MIXED_STRATEGY_GAMES:
				skip = discardNonMixedStrategyGames(eq)
			elif FILTER_PURE_STRATEGIES: #If enabled, filter out pure strategy equilibria
				eq, skip = filterPureStrategies(eq)
			
			if skip:
				continue
			
			#If it got here, game is not degenerate
			games[count] = g

			#Set the number of Nash equilibria to a predefined one
			nash_nSet = convertToN(eq, max_nashes = max_nashes, players = players, strategies = strategies)

			#Store nash in nashes
			nashes[count] = nash_nSet

			#Increment Count
			count = count + 1
			
			#Re-enable printing a status
			printed = False

		except RuntimeWarning:
			num_degen = num_degen + 1
# 			print("Error - game is degenerate")

	#Print
	print("Generated {} games in {} seconds with {} degenerate games.".format(count, time.time() - start, num_degen))

	#Return games, nashes
	return games, nashes

# d = {"Game":g,"Nash1-SuppEnum":sup_enum,"Nash2-LH":lemke_howson, "Nash3-VertEnum":vertex_enum}
# df = pd.DataFrame(d)

# df.to_csv("million-val.csv")


# print(num_degen)
g, n = generate_dataset(num_games = NUMBER_OF_SAMPLES)
np.save("./Datasets/" + GENERATED_GAMES_DATASET_NAME, g)
np.save("./Datasets/" + GENERATED_EQUILIBRIA_DATASET_NAME, n)

