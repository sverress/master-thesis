import decision
import random
import globals


def get_policy(name="RandomRolloutPolicy"):
    if name == "RandomRolloutPolicy":
        return decision.RandomRolloutPolicy()
    elif name == "SwapAllPolicy":
        return decision.SwapAllPolicy()
    else:
        raise ValueError("No current policy name were given")


def epsilon_greedy(choices: [(float, object)], epsilon=globals.EPSILON):
    """
    :param choices: a list of tuples with values associated with a object
    :param epsilon: the probability of choosing a random object rather than the highest
    :return: Either best or random object of the input choices
    """
    if random.random() > epsilon:
        chosen = max(choices, key=lambda choice: choice[0])
    else:
        chosen = random.choice(choices)
    value, chosen_object = chosen
    return chosen_object
