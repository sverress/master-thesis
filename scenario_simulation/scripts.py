import copy
import numpy as np
from classes.State import State
from globals import ITERATION_LENGTH_MINUTES, LOST_TRIP_REWARD


def estimate_reward(
    state: State, remaining_shift_duration: int, number_of_simulations=10
):
    """
    Does n times scenario simulations and returns the highest conducted reward from simulation
    :param state: State - state to de the simulations from
    :param remaining_shift_duration: int - time left on shift = length of simulation
    :param number_of_simulations: int - number of simulations to be performed (default = 10)
    :return: int - maximum reward from simulations
    """

    all_rewards = []

    # Do n scenario simulations
    for i in range(number_of_simulations):
        iteration_counter = 0
        child_state = copy.deepcopy(state)
        total_reward = 0

        # Simulate until shift ends
        while iteration_counter * ITERATION_LENGTH_MINUTES < remaining_shift_duration:
            iteration_counter += 1
            # all possible actions in this state
            possible_actions = child_state.get_possible_actions(number_of_neighbours=3)

            # pick a random action
            random_action = possible_actions[
                np.random.randint(0, len(possible_actions))
            ]
            total_reward += child_state.do_action(random_action)

            _, _, lost_demand = child_state.system_simulate()

            total_reward -= lost_demand * LOST_TRIP_REWARD

        all_rewards.append(total_reward)

    return max(all_rewards)
