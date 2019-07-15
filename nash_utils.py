import random
import numpy as np
import pandas as pd
import nashpy as nash
import itertools, time

import warnings
warnings.filterwarnings("error")

def generate_game(players=2, size=(3,3)):
	game = np.zeros((players, size[0],size[1]))
	for i in range(players):
		game[i] = np.random.rand(size[0],size[1])
	return game

def generate_nash(game):
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


def ConvertToTen(array, max_nashes=10, players=2, options=3):
	#Create numpy array to store nashes
	nash = np.zeros((max_nashes, players, options))
    
	#Create itertools cycle
	cycle = itertools.cycle(array)

	#Iterate through list and indices
	for i, elem in zip(range(max_nashes), cycle):
		nash[i] = np.array(elem)

	return nash


def generate_dataset(num_games=1000, max_nashes=10, players=2, options=3):
	# Create an empty numpy array with proper shape for games and nashes
	games = np.zeros((num_games, players, options, options))
	nashes = np.zeros((num_games, max_nashes, players, options))

	# Loop
	start = time.time()
	count = 0
	num_degen = 0
	while( count < num_games):
		if count%250 == 0:
			print("Generated ",count," games in ", time.time()-start, 
			 	" seconds with ", num_degen, " degenerate games.")
		#Put game generation in a try statement and catch runtime warning if
		#	the game is degenerate, retry without incrementing count
		try:
			# Generate a game
			g = generate_game(players=players, size=(options, options))

			#Get the nash equilibria
			n1, n2, n3 = generate_nash(g)

			#If it got here, game is not degenerate
			games[count] = g

			#Convert n nashes to 10 nashes
			nash = ConvertToTen(n1, max_nashes=max_nashes, 
									players=players, options=options)

			#Store nash in nashes
			nashes[count] = nash

			#Increment Count
			count = count+1

		except RuntimeWarning:
			num_degen = num_degen + 1
			print("Error - game is degenerate")

	#Print
	print("Generated ",count," games in ", time.time()-start, 
		  " seconds with ", num_degen, " degenerate games.")

	#Return games, nashes
	return games, nashes

# g=[]
# n=[]
# sup_enum=[]
# lemke_howson=[]
# vertex_enum=[]
# num_degen = 0
# for i in range(0,250000):
# 	try:
# 		g1 = generate_game()
# 		n1, n2, n3 = generate_nash(g1)
# 		g.append(g1)
# 		sup_enum.append(n1)
# 		lemke_howson.append(n2)
# 		vertex_enum.append(n3)
# 		# n.append(n1)
# 	except RuntimeWarning:
# 		num_degen = num_degen + 1
# 		continue
# d = {"Game":g,"Nash1-SuppEnum":sup_enum,"Nash2-LH":lemke_howson, "Nash3-VertEnum":vertex_enum}
# df = pd.DataFrame(d)

# df.to_csv("million-val.csv")

# print(num_degen)
g, n = generate_dataset(num_games=50000000)
np.save("50M", g)
np.save("50M-Labels", n)

# for i in range(n.shape[0]):
# 	print(n[i])
# print("================")
# print(n)
# print(type(n))
# print("================")
# print(n[0])
# print(type(n[0]))
# print("================")
# print(n[0][0])
# print(type(n[0][0]))