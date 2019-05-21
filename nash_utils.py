import random
import numpy as np
import pandas as pd
import nashpy as nash

import warnings
warnings.filterwarnings("error")

def generate_game(size=(3,3)):
	A = np.random.rand(3,3)
	B = np.random.rand(3,3)
	return (A,B)

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

df = pd.DataFrame(columns=["Game","Nash"])
dict_list = []
num_degen = 0
for i in range(0,1000):
	try:
		g = generate_game()
		n, _, _ = generate_nash(g)
		d = {"Game":g,"Nash":n}
		dict_list.append(d)
	except RuntimeWarning:
		num_degen = num_degen + 1
		continue
df2 = df.append(dict_list)
df2.to_csv("test.csv")

print(num_degen)
