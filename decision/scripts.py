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
    Improvements: - Travel only to neighbouring clusters.
    - Only perform some battery swaps, eg. even numbers
    :param state: current state, State
    :return: List of object Action
    """
    actions = []

    current_cluster = state.current
    # Assume that no battery swap or pick-up of scooter with 100% battery and
    # that the scooters with the lowest battery are swapped and picked up
    swappable_scooters = current_cluster.get_swappable_scooters()

    # Different combinations of battery swaps, pick-ups, drop-offs and clusters
    for cluster in state.clusters:
        # Next cluster cant be same as current
        if cluster == current_cluster:
            continue
        # Edge case: Add action with no swap, pick-up or drop-off
        actions.append(Action([], [], [], cluster))
        if current_cluster.number_of_possible_pickups() == 0:
            # Battery swap and drop-off
            battery_counter = 0
            while (
                battery_counter < state.vehicle.battery_inventory
                and battery_counter < len(swappable_scooters)
            ):
                # Edge case: No drop-offs, but all combinations of battery swaps and clusters.
                actions.append(
                    Action(
                        [swappable_scooters[i] for i in range(battery_counter + 1)],
                        [],
                        [],
                        cluster,
                    )
                )
                battery_counter += 1
            delivery_counter = 0
            while (
                delivery_counter < len(state.vehicle.scooter_inventory)
                and delivery_counter + current_cluster.number_of_scooters()
                < current_cluster.ideal_state
            ):
                # Edge case: No pick-ups and battery swaps, but all combinations of delivery scooters and next cluster
                actions.append(
                    Action(
                        [],
                        [],
                        [
                            state.vehicle.scooter_inventory[i]
                            for i in range(delivery_counter + 1)
                        ],
                        cluster,
                    )
                )
                # All possible battery swap combinations, combined with drop-off and next cluster combinations
                battery_counter = 0
                while (
                    battery_counter < state.vehicle.battery_inventory
                    and battery_counter < len(swappable_scooters)
                ):
                    actions.append(
                        Action(
                            [swappable_scooters[i] for i in range(battery_counter + 1)],
                            [],
                            [
                                state.vehicle.scooter_inventory[i]
                                for i in range(delivery_counter + 1)
                            ],
                            cluster,
                        )
                    )
                    battery_counter += 1

                delivery_counter += 1

        # Battery swap and pick-up
        else:

            battery_counter = 0
            while (
                battery_counter < state.vehicle.battery_inventory
                and battery_counter < len(swappable_scooters)
            ):
                # Edge case: No pick-ups, but all combinations of battery swaps and clusters.
                actions.append(
                    Action(
                        [swappable_scooters[i] for i in range(battery_counter + 1)],
                        [],
                        [],
                        cluster,
                    )
                )
                battery_counter += 1

            pick_up_counter = 0
            while (
                pick_up_counter < state.vehicle.battery_inventory
                and pick_up_counter < len(swappable_scooters)
                and pick_up_counter
                < (len(current_cluster.scooters) - current_cluster.ideal_state)
            ):
                # Edge case: No battery swaps, but all combinations of pick-ups and clusters.
                actions.append(
                    Action(
                        [],
                        [swappable_scooters[i] for i in range(pick_up_counter + 1)],
                        [],
                        cluster,
                    )
                )
                # Combinations of battery swaps, pick-ups and clusters
                # Pick up the scooters with lowest battery, swap the next lowest.
                battery_counter = pick_up_counter + 1
                while (
                    battery_counter < state.vehicle.battery_inventory
                    and battery_counter < len(swappable_scooters)
                ):
                    actions.append(
                        Action(
                            [
                                swappable_scooters[i]
                                for i in range(pick_up_counter + 1, battery_counter + 1)
                            ],
                            [swappable_scooters[i] for i in range(pick_up_counter + 1)],
                            [],
                            cluster,
                        )
                    )
                    battery_counter += 1
                pick_up_counter += 1

    return actions


def get_action_time(action: Action, state: State):
    """
    Get the time consumed from performing an action (travel from cluster 1 to 2) in a given state.
    Can add time for performing actions on scooters as well.
    :param action: Action to be performed
    :param state: current state, State
    :return: Total time to perform action
    """
    return 0.25


def get_current_reward(action: Action, state: State):
    return


def run(duration):
    """

    :param duration: shift time in minutes
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
            remaining_duration = duration - (t + get_action_time(new_state, action))
            reward = estimate_reward(
                new_state, remaining_duration
            ) + get_current_reward(action, state)

            if reward > max_reward:
                max_reward = reward
                best_action = action

        all_actions.append(best_action)
        t += get_action_time(state, best_action)

        # Perform best action
        reward = state.do_action(best_action)
        total_reward += reward

        # System simulation
        system_simulate(state)

    return total_reward, all_actions


if __name__ == "__main__":
    state = get_initial_state()
    state.vehicle.scooter_inventory = state.current.scooters[7:9]
    state.current.scooters = state.current.scooters[:5]
    state.clusters = state.clusters[1:3]
    state.current.ideal_state = 6

    print(f"Scooter inventory: {len(state.vehicle.scooter_inventory)}")
    print(f"Battery inventory: {state.vehicle.battery_inventory}")
    print(f"Number of scooters in cluster: {len(state.current.scooters)}")
    print(f"Ideal state in cluster: {state.current.ideal_state}")
    print(f"Number of possible next clusters: {len(state.clusters)}")

    actions = get_possible_actions(state)
    print("All possible actions:")
    for action in actions:
        print(
            f"Battery swap: {len(action.battery_swaps)}, Pick-ups: {len(action.pick_ups)},"
            f" Deliveries: {len(action.delivery_scooters)}, Next cluster: {action.next_cluster} "
        )

    print(f"Total number of actions: {len(actions)}")
