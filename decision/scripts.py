import copy

from clustering.scripts import get_initial_state
from classes.Action import Action
from classes.State import State
from scenario_simulation.scripts import estimate_reward
from system_simulation.scripts import system_simulate


def get_state(action: Action, state: State):
    """
    Returns the updated state after performing a given action.
    :param action: Action to be performed
    :param state: Current state, deepcopy
    :return: new state State after performing action Action
    """
    # Move to new next cluster
    state.current = action.next_cluster

    # Perform pickups
    for pick_up in action.pick_ups:
        state.vehicle.pick_up(pick_up)

    # Perform battery change
    state.vehicle.change_batteries(action.battery_swaps)

    # Perform drop-off
    for delivery_scooter in action.delivery_scooters:
        state.vehicle.drop_off(delivery_scooter)

    return state


def run(duration):
    """
    :param duration: shift time in minutes
    :return: Total reward of 1 shift, list of all actions taken
    """

    t = 0
    total_reward = 0
    all_actions = []
    # Get data from database
    state = get_initial_state()

    while t < duration:
        max_reward = 0
        best_action = None

        # Find all possible actions
        actions = state.get_possible_actions()

        # For every possible action
        for action in actions:
            # Get new state of performing action
            new_state = get_state(action, copy.deepcopy(state))

            # Estimate value of making this action
            next_cluster_distance = state.get_distance(
                state.current_cluster, action.next_cluster
            )
            remaining_duration = duration - (
                t + action.get_action_time(next_cluster_distance)
            )
            reward = estimate_reward(
                new_state, remaining_duration
            ) + state.get_current_reward(action)

            if reward > max_reward:
                max_reward = reward
                best_action = action

        all_actions.append(best_action)
        best_next_cluster_distance = state.get_distance(
            state.current_cluster, best_action.next_cluster
        )
        t += best_action.get_action_time(best_next_cluster_distance)

        # Perform best action
        reward = state.do_action(best_action)
        total_reward += reward

        # System simulation
        system_simulate(state)

    return total_reward, all_actions


if __name__ == "__main__":

    # Create an initial state. Smaller than normal for testing purposes
    state = get_initial_state()
    state.vehicle.scooter_inventory = state.current_cluster.scooters[7:9]
    state.current_cluster.scooters = state.current_cluster.scooters[:5]
    state.clusters = state.clusters[1:2]
    state.current_cluster.ideal_state = 7

    # Print the initial state
    print(f"Scooter inventory: {len(state.vehicle.scooter_inventory)}")
    print(f"Battery inventory: {state.vehicle.battery_inventory}")
    print(f"Number of scooters in cluster: {len(state.current_cluster.scooters)}")
    print(f"Ideal state in cluster: {state.current_cluster.ideal_state}")
    print(f"Number of possible next clusters: {len(state.clusters)}")

    # Get all possible actions.
    actions = state.get_possible_actions()
    print("All possible actions:")
    for action in actions:
        print(
            f"Battery swap: {len(action.battery_swaps)}, Pick-ups: {len(action.pick_ups)},"
            f" Deliveries: {len(action.delivery_scooters)}, Next cluster: {action.next_cluster} "
        )
    print(f"Total number of actions: {len(actions)}")

    """
    # Testing get_action_time
    best_action = actions[21]
    next_cluster_distance = state.get_distance(
        state.current_cluster, best_action.next_cluster
    )
    print(
        f"Time to perform action: {best_action.get_action_time(next_cluster_distance)}"
    )
    """
