import copy

from clustering.scripts import get_initial_state
from classes.Action import Action
from classes.State import State
from scenario_simulation.scripts import estimate_reward
from system_simulation.scripts import system_simulate


def run(duration):
    """
    :param duration: shift time in minutes
    :return: Total reward of 1 shift, list of all actions taken
    """

    # Init amount of time used, total reward and list of actions taken
    shift_duration_used = 0
    total_reward = 0
    all_actions = []

    # Get data from database
    state = get_initial_state(sample_size=100, number_of_clusters=10)

    while shift_duration_used < duration:
        max_reward = 0
        best_action = None
        best_next_cluster_distance = 0
        print(shift_duration_used)

        # Find all possible actions
        actions = state.get_possible_actions(number_of_neighbours=3)

        # For every possible action
        for action in actions:
            # Get new state of performing action
            new_state = copy.deepcopy(state)
            reward = new_state.do_action(action)

            # Get distance to next_cluster in action
            next_cluster_distance = state.get_distance(
                state.current_cluster, action.next_cluster
            )
            # Calculate remaining duration of shift after performing action
            remaining_duration = duration - (
                shift_duration_used + action.get_action_time(next_cluster_distance)
            )
            # Estimate value of making this action, after performing it and calculating the time it takes to perform.
            reward += estimate_reward(new_state, remaining_duration)

            # If the action is better than previous actions, make best_action
            # Add next cluster distance to update shift duration used later.
            if reward >= max_reward:
                max_reward = reward
                best_action = action
                best_next_cluster_distance = next_cluster_distance

        # Add best action to actions taken, and update shift duration used.
        all_actions.append(best_action)
        shift_duration_used += best_action.get_action_time(best_next_cluster_distance)

        # Perform best action
        total_reward += state.do_action(best_action)

        # System simulation
        # TODO This is only to happen every 20 minutes.
        system_simulate(state)

    return total_reward, all_actions
