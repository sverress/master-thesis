import copy
import numpy as np
from classes.State import State


def estimate_reward(state: State, remaining_shift_duration: int):

    # Overall purpose -> do n random simulations from this state until shift ends
    number_of_simulations = 10
    length_of_iteration = 20
    all_rewards = []

    for i in range(number_of_simulations):
        iteration_counter = 0
        initial_state = copy.deepcopy(state)
        while iteration_counter * length_of_iteration < remaining_shift_duration:
            iteration_counter += 1
            actions = state.get_possible_actions()
            random_action = actions[np.random.randint(0, len(actions))]

    # For at timestep -> get all actions, pick a random action
    # When shift ends, vehicle has to return to the depot (later implementation)
    # Repeat until shift ends

    # Repeat for n simulations for original state

    # Return average of all rewards
    # Perform that action on the state and record the reward
    # Do a Markov Decision Process

    return 10
