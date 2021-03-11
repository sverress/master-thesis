from scenario_simulation.scripts import estimate_reward
from classes import State, Action
import copy


def get_best_action(state: State, remaining_time: int) -> Action:

    max_reward = 0
    best_action = None

    # Find all possible actions
    actions = state.get_possible_actions(number_of_neighbours=3, divide=2)

    # For every possible action
    for action in actions:
        # Get new state of performing action
        new_state = copy.deepcopy(state)
        reward = new_state.do_action(action)

        # Estimate value of making this action, after performing it and calculating the time it takes to perform.
        reward += estimate_reward(new_state, remaining_time)

        # If the action is better than previous actions, make best_action
        # Add next cluster distance to update shift duration used later.
        if reward >= max_reward:
            max_reward = reward
            best_action = action

    return best_action
