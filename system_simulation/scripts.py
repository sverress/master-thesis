import numpy as np


def system_simulate(state):
    """
    Simulation of poisson process on the system ->
    Poisson distributed number of trips out of each cluster, markov chain decides where the trip goes
    :param state: current state
    :return: new state after a simulation of the poisson process
    """
    flow_counter = {
        (start, end): 0
        for start in np.arange(len(state.clusters))
        for end in np.arange(len(state.clusters))
        if start != end
    }
    min_battery = 20.0
    trips = []
    for i, start_cluster in enumerate(state.clusters):
        # poisson process to select number of trips in a iteration
        number_of_trips = round(
            np.random.poisson(start_cluster.trip_intensity_per_iteration)
        )

        # can't complete more trips then there is scooters with battery over min_battery
        valid_scooters = start_cluster.get_valid_scooters(min_battery)
        if number_of_trips > len(valid_scooters):
            number_of_trips = len(valid_scooters)

        # collect n neighbours for the cluster (can be implemented with distance limit)
        neighbours = state.get_neighbours(start_cluster, number_of_neighbours=3)

        # make the markov chain out of the cluster
        prob_distribution = [start_cluster.prob_leave(neigh) for neigh in neighbours]

        normalized_prob_distribution = np.true_divide(
            prob_distribution, sum(prob_distribution)
        )

        # loop to generate trips from the cluster
        for j in range(number_of_trips):
            end_cluster = neighbours[
                np.random.choice(
                    np.arange(len(normalized_prob_distribution)),
                    p=normalized_prob_distribution,
                )
            ]

            trips.append((start_cluster, end_cluster, valid_scooters[j]))
            flow_counter[(start_cluster.id, end_cluster.id)] += 1

    # compute trip after all trips are generated to avoid handling inflow in cluster
    for start_cluster, end_cluster, scooter in trips:
        start_cluster.scooters.remove(scooter)
        trip_distance = state.get_distance(start_cluster, end_cluster)
        scooter.travel(trip_distance)
        end_cluster.add_scooter(scooter)

    return (
        [
            (start_end[0], start_end[1], flow)
            for start_end, flow in list(flow_counter.items())
        ],
        trips,
    )
