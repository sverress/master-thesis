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

    remaining_time = 0
    total_reward = 0
    all_actions = []
    # Get data from database
    state = get_initial_state()

    while remaining_time < duration:
        max_reward = 0
        best_action = None

        # Find all possible actions
        actions = state.get_possible_actions(number_of_neighbours=3)
        # TODO next cluster is index of next cluster, not the object itself.

        # For every possible action
        for action in actions:
            # Get new state of performing action
            new_state = copy.deepcopy(state)
            new_state.current_cluster = new_state.clusters[
                state.clusters.index(state.current_cluster)
            ]
            reward = new_state.do_action(action)

            # Estimate value of making this action
            next_cluster_distance = state.get_distance(
                state.current_cluster, action.next_cluster
            )
            remaining_duration = duration - (
                remaining_time + action.get_action_time(next_cluster_distance)
            )
            reward += estimate_reward(new_state, remaining_duration)

            if reward > max_reward:
                max_reward = reward
                best_action = action

        all_actions.append(best_action)
        best_next_cluster_distance = state.get_distance(
            state.current_cluster, best_action.next_cluster
        )
        remaining_time += best_action.get_action_time(best_next_cluster_distance)

        # Perform best action
        total_reward += state.do_action(best_action)

        # System simulation
        # TODO This is only to happen every 20 minutes.
        system_simulate(state)

    return total_reward, all_actions


run(480)
