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
            possible_actions = child_state.get_possible_actions(number_of_neighbours=3)

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

    # Generate scooter moves
    trips = (
        []
    )  # (start_cluster: Cluster, end_cluster: Cluster, scooter: Scooter, distance: int)

    # generate all trips
    for cluster in state.clusters:
        # collect n neighbours for the cluster (can be implemented with distance limit)
        neighbours = state.get_neighbours(cluster, number_of_neighbours=3)

        # make the markov chain out of the cluster (includes probability of staying in the cluster)
        prob_distribution = [cluster.prob_stay()] + [
            cluster.prob_leave(neigh) for neigh in neighbours
        ]

        # for all scooters in the cluster -> perform a trip to another cluster or stay
        for scooter in cluster.scooters:
            # pick a random destination or stay
            destination = np.random.choice(
                np.arange(len(prob_distribution)), p=prob_distribution
            )
            if destination != 0:
                trips.append(
                    (
                        state.current_cluster,
                        state.clusters[destination],
                        scooter,
                        state.get_distance(
                            state.current_cluster, state.clusters[destination]
                        ),
                    )
                )

    # perform all trips
    for start, end, scooter, distance in trips:
        start.remove(scooter)
        end.add_scooter(scooter)
        scooter.travel(distance)
