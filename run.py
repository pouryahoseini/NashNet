import tensorflow as tf
from NashNet import NashNet

q = NashNet("./test_config.cfg")
# q.add_model("./test/saved_weights/model0/model0final_weights.h5")
for i in range(20):
    q.add_model()
    # q.add_model("./test/saved_weights/model0/model0final_weights.h5")
q.train((["Dataset-DiscardedPure-1_Games_2P-3x3_1M.npy.npy"], ["Dataset-DiscardedPure-1_Equilibria_2P-3x3_1M.npy"]))
# q.evaluate((["Dataset-DiscardedPure-1_Games_2P-3x3_1M.npy.npy"], ["Dataset-DiscardedPure-1_Equilibria_2P-3x3_1M.npy"]))

print("Pie!")