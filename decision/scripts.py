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


def get_possible_actions(state: State):
    """
    Need to figure out what actions we want to look at. The combination of pick-ups, battery swaps,
    drop-offs and next cluster is too large to try them all.
    :param state: current state, State
    :return: List of object Action
    """
    actions = []

    current_cluster = state.current




    # Delivery scooters and clusters
    for cluster in state.clusters:
        # Add action with no drop-off
        actions.append(Action([], [], [], cluster))
        if cluster.number_of_possible_pickups() == 0:
            # for delivery_scooter in state.vehicle.scooter_inventory:
            delivery_counter = 0
            while (delivery_counter < state.vehicle.scooter_inventory and
                   delivery_counter + current_cluster.number_of_scooters() <= current_cluster.ideal_state):
                # No pick-ups and battery swaps, but all combinations of delivery scooters and next cluster
                actions.append(Action([],
                                      [],
                                      [state.vehicle.scooter_inventory[i] for i in range(delivery_counter + 1)],
                                      cluster))

        else:
            for scooter in current_cluster.scooters:
                pass

    return actions


def get_action_time(action: Action, state: State):
    """
    Get the time consumed from performing an action (travel from cluster 1 to 2) in a given state.
    Can add time for performing actions on scooters as well.
    :param action: Action to be performed
    :param state: current state, State
    :return: Total time to perform action
    """
    pass


def get_current_reward(action: Action, state: State):
    pass


def run(duration):
    """

    :param duration: shift time in hours
    :return:
    """
    # Se hva som er nåværende state.
    # For et visst antall ekstreme lovlige actions, kjøre scenario simulering
    # for alle naboer (Hvordan definere nabo? Alle clusters?)
    # Velge neste node for en gitt action.
    # Sørge for at det legges til tid å reise et sted, kanskje for å utføre bytter også?
    # Reward er reward for neste cluster og beslutningen tatt i current state.
    # Pickup reward ikke i current, men drop-off er det.
    # Velge neste node og action i current cluster som gir høyest reward.

    t = 0
    total_reward = 0
    all_actions = []
    # Get data from database
    state = get_initial_state()

    while t < duration:
        max_reward = 0
        best_action = None

        # Find all possible actions
        actions = get_possible_actions(state)

        # For every possible action
        for action in actions:
            # Get new state of performing action
            new_state = get_state(action, copy.deepcopy(state))

            # Estimate value of making this action
            reward = estimate_reward(new_state) + get_current_reward(action, state)

            if reward > max_reward:
                max_reward = reward
                best_action = action

        total_reward += max_reward
        all_actions.append(best_action)
        t += get_action_time(state, best_action)
        # System simulation
        state = system_simulate(best_action)

    return total_reward, all_actions


run(8)
