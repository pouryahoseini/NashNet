import numpy as np

# All games
NUMBER_OF_SAMPLES = 100
RANDOM_NUMBER_RANGE = 100

# Coordination Game
NUMBER_OF_PLAYERS_COORDINATION = 3
STRATEGIES_NUMBER_COORDINATION = 2

# Volunteer's Dilemma
NUMBER_OF_PLAYERS_VOLUNTEER = 3


# *****************************
def coordination_game(number_of_samples=NUMBER_OF_SAMPLES, random_number_range=RANDOM_NUMBER_RANGE, number_of_players=NUMBER_OF_PLAYERS_COORDINATION, strategies_number=STRATEGIES_NUMBER_COORDINATION, data_type=np.float64):
    games = np.zeros((number_of_samples, number_of_players) + (strategies_number,) * number_of_players, dtype=data_type)

    for sample in range(number_of_samples):
        for player in range(number_of_players):
            for strategy in range(strategies_number):
                top_utility = np.random.randint(1, random_number_range)

                idx = [sample, player] + [slice(strategies_number)] * number_of_players
                for players_index in range(number_of_players):
                    if players_index != player:
                        idx_copy = idx + []
                        idx_copy[2 + players_index] = strategy
                        games[tuple(idx_copy)] = np.random.randint(top_utility)

                idx_2 = (sample, player) + (strategy,) * number_of_players
                games[idx_2] = top_utility

    return games


# *****************************
def volunteer_dilemma(number_of_samples=NUMBER_OF_SAMPLES, random_number_range=RANDOM_NUMBER_RANGE, number_of_players=NUMBER_OF_PLAYERS_VOLUNTEER, data_type=np.float64):
    games = np.zeros((number_of_samples, number_of_players) + (2,) * number_of_players, dtype=data_type)

    for sample in range(number_of_samples):
        for player in range(number_of_players):
            not_cooperate_others_do_utility = np.random.randint(2, random_number_range)
            cooperate_utility = np.random.randint(1, not_cooperate_others_do_utility)
            no_one_cooperates_utility = np.random.randint(cooperate_utility)

            idx = [sample, player] + [slice(2)] * number_of_players

            idx[2 + player] = 0
            games[tuple(idx)] = cooperate_utility

            idx[2 + player] = 1
            games[tuple(idx)] = not_cooperate_others_do_utility

            idx = [sample, player] + [1] * number_of_players
            games[tuple(idx)] = no_one_cooperates_utility

    return games


# *****************************
def prisoner_dilemma(number_of_samples=NUMBER_OF_SAMPLES, random_number_range=RANDOM_NUMBER_RANGE, data_type=np.float64):
    games = np.zeros((number_of_samples, 2, 2, 2), dtype=data_type)

    for sample in range(number_of_samples):
        pl_1_top_left = np.random.randint(2, random_number_range - 1)

        games[sample, 0, 0, 0] = pl_1_top_left
        games[sample, 1, 0, 0] = pl_1_top_left

        pl_1_bottom_right = np.random.randint(1, pl_1_top_left)
        games[sample, 0, 1, 1] = pl_1_bottom_right
        games[sample, 1, 1, 1] = pl_1_bottom_right

        pl_1_bottom_left = np.random.randint(pl_1_top_left + 1, random_number_range)
        games[sample, 0, 1, 0] = pl_1_bottom_left
        games[sample, 1, 0, 1] = pl_1_bottom_left

        games[sample, 1, 1, 0] = 0
        games[sample, 0, 0, 1] = 0

    return games
