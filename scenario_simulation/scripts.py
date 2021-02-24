import copy
import numpy as np
from classes.State import State


def estimate_reward(
    state: State, remaining_shift_duration: int, number_of_simulations=10
):
    """
    Does n times scenario simulations and returns the highest conducted reward from simulation
    :param state: State - state to de the simulations from
    :param remaining_shift_duration: int - time left on shift = length of simulation
    :param number_of_simulations: int - number of simulations to be performed (default = 10)
    :return: int - maximum reward of simulations
    """

    number_of_simulations = 10
    length_of_iteration = 20
    all_rewards = []

    # Do n scenario simulations
    for i in range(number_of_simulations):
        iteration_counter = 0
        child_state = copy.deepcopy(state)
        total_reward = 0

        # Simulate until shift ends
        while iteration_counter * length_of_iteration < remaining_shift_duration:
            iteration_counter += 1
            # all possible actions in this state
            possible_actions = child_state.get_possible_actions()

            # pick a random action
            random_action = possible_actions[
                np.random.randint(0, len(possible_actions))
            ]
            total_reward += child_state.do_action(random_action)

            # TODO Do a Markov Decision Process from markov_decision_process()

        all_rewards.append(total_reward)

    return max(all_rewards)


def markov_decision_process(state: State):
    """
    :param state: State - current state to perform one iteration of the markov decision process
    """
    raise NotImplementedError("This function is not yet implemented")
    # Initialize trips. Trip format: (start_cluster: Cluster, end_cluster: Cluster, scooter: Scooter, distance: int)
    trips = []
    # Generate scooter trips
    for cluster in state.clusters:
        # For all scooters in the cluster -> perform a trip to another cluster or stay
        for scooter in cluster.scooters:
            # Pick a random destination
            destination = np.random.choice(
                sorted(state.clusters, key=lambda state_cluster: state_cluster.id),
                p=cluster.get_leave_distribution(),
            )
            trips.append(
                (
                    state.current_cluster,
                    destination,
                    scooter,
                    state.get_distance(state.current_cluster, destination),
                )
            )

    # perform all trips
    for start, end, scooter, distance in trips:
        start.remove_scooter(scooter)
        end.add_scooter(scooter)
        scooter.travel(distance)
