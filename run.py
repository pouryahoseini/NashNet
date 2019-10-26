from NashNet import NashNet
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
q = NashNet("./test_config.cfg")
# q.add_model("./test/saved_weights/model0/model0final_weights.h5")
weights = ["./best_model_weights/model-0-weights.h5",
           "./best_model_weights/model-1-weights.h5",
           "./best_model_weights/model-2-weights.h5",
           "./best_model_weights/model3final_weights.h5",
           "./best_model_weights/model4final_weights.h5"]
# for i in range(20):
#     q.add_model()
# for w in weights:
#     q.add_model(w)
# q.train_new_model((["Dataset-DiscardedPure-1_Games_2P-3x3_1M.npy.npy"], ["Dataset-DiscardedPure-1_Equilibria_2P-3x3_1M.npy"]), weights=weights[0])
# q.evaluate_models(data_files=(["Dataset-DiscardedPure-1_Games_2P-3x3_1M.npy.npy"], ["Dataset-DiscardedPure-1_Equilibria_2P-3x3_1M.npy"]),
#                   weights=weights)
q.evaluate_models(data_files=(["Dataset-DiscardedPure-2_Games_2P-3x3_1M.npy.npy"], ["Dataset-DiscardedPure-2_Equilibria_2P-3x3_1M.npy"]),
                  weights=weights)

print("Pie!")

