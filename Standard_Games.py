import numpy as np

NUMBER_OF_SAMPLES = 100
RANDOM_NUMBER_RANGE = 100

NUMBER_OF_PLAYERS_COORDINATION = 3
STRATEGIES_NUMBER_COORDINATION = 2

NUMBER_OF_PLAYERS_VOLUNTEER = 3


# *****************************
def coordination_game():
    games = np.zeros((NUMBER_OF_SAMPLES, NUMBER_OF_PLAYERS_COORDINATION) + (STRATEGIES_NUMBER_COORDINATION,) * NUMBER_OF_PLAYERS_COORDINATION, dtype=np.float64)

    for sample in range(NUMBER_OF_SAMPLES):
        for player in range(NUMBER_OF_PLAYERS_COORDINATION):
            for strategy in range(STRATEGIES_NUMBER_COORDINATION):
                top_utility = np.random.randint(1, RANDOM_NUMBER_RANGE)

                idx = [sample, player] + [slice(STRATEGIES_NUMBER_COORDINATION)] * NUMBER_OF_PLAYERS_COORDINATION
                for players_index in range(NUMBER_OF_PLAYERS_COORDINATION):
                    if players_index != player:
                        idx_copy = idx + []
                        idx_copy[2 + players_index] = strategy
                        games[tuple(idx_copy)] = np.random.randint(top_utility)

                idx_2 = (sample, player) + (strategy,) * NUMBER_OF_PLAYERS_COORDINATION
                games[idx_2] = top_utility

    return games


# *****************************
def volunteer_dilemma():
    games = np.zeros((NUMBER_OF_SAMPLES, NUMBER_OF_PLAYERS_VOLUNTEER) + (2,) * NUMBER_OF_PLAYERS_VOLUNTEER, dtype=np.float64)

    for sample in range(NUMBER_OF_SAMPLES):
        for player in range(NUMBER_OF_PLAYERS_VOLUNTEER):
            not_cooperate_others_do_utility = np.random.randint(2, RANDOM_NUMBER_RANGE)
            cooperate_utility = np.random.randint(1, not_cooperate_others_do_utility)
            no_one_cooperates_utility = np.random.randint(cooperate_utility)

            idx = [sample, player] + [slice(2)] * NUMBER_OF_PLAYERS_VOLUNTEER

            idx[2 + player] = 0
            games[tuple(idx)] = cooperate_utility

            idx[2 + player] = 1
            games[tuple(idx)] = not_cooperate_others_do_utility

            idx = [sample, player] + [1] * NUMBER_OF_PLAYERS_VOLUNTEER
            games[tuple(idx)] = no_one_cooperates_utility

    return games


# *****************************
def prisoner_dilemma():
    games = np.zeros((NUMBER_OF_SAMPLES, 2, 2, 2), dtype=np.float64)

    for sample in range(NUMBER_OF_SAMPLES):
        pl_1_top_left = np.random.randint(2, RANDOM_NUMBER_RANGE - 1)

        games[sample, 0, 0, 0] = pl_1_top_left
        games[sample, 1, 0, 0] = pl_1_top_left

        pl_1_bottom_right = np.random.randint(1, pl_1_top_left)
        games[sample, 0, 1, 1] = pl_1_bottom_right
        games[sample, 1, 1, 1] = pl_1_bottom_right

        pl_1_bottom_left = np.random.randint(pl_1_top_left + 1, RANDOM_NUMBER_RANGE)
        games[sample, 0, 1, 0] = pl_1_bottom_left
        games[sample, 1, 0, 1] = pl_1_bottom_left

        games[sample, 1, 1, 0] = 0
        games[sample, 0, 0, 1] = 0

    return games
