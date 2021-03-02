import copy

from clustering.scripts import get_initial_state
from scenario_simulation.scripts import estimate_reward
from system_simulation.scripts import system_simulate
from visualization.visualizer import *


def run(
    duration,
    sample_size=100,
    number_of_clusters=10,
    scenario_iterations=10,
    visualize=False,
):
    """
    :param duration: shift time in minutes
    :param sample_size: number of scooters in he system
    :param number_of_clusters: number of clusters in the system
    :param scenario_iterations: number of iterations per scenario simulation ( 1 itt = 1 random roll out of state)
    :param visualize: if the solution of system where to be visualised
    :return: Total reward of 1 shift, list of all actions taken
    """

    # Init amount of time used, total reward and list of actions taken
    shift_duration_used = 0
    total_reward = 0
    all_actions = []

    # Get data from database
    state = get_initial_state(
        sample_size=sample_size, number_of_clusters=number_of_clusters
    )

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
            reward += estimate_reward(
                new_state, remaining_duration, scenario_iterations
            )

            # If the action is better than previous actions, make best_action
            # Add next cluster distance to update shift duration used later.
            if reward >= max_reward:
                max_reward = reward
                best_action = action
                best_next_cluster_distance = next_cluster_distance

        # Add best action to actions taken, and update shift duration used.
        all_actions.append(best_action)
        shift_duration_used += best_action.get_action_time(best_next_cluster_distance)

        if visualize:
            previous_state = copy.deepcopy(state)

        # Perform best action
        total_reward += state.do_action(best_action)

        # System simulation
        # TODO This is only to happen every 20 minutes.
        flows, trips = system_simulate(state)

        if visualize:
            previous_state.visualize()
            previous_state.visualize_flow(flows, best_action.next_cluster.id)
            previous_state.visualize_action(state, best_action)
            previous_state.visualize_system_simulation(trips)

    return total_reward, all_actions
