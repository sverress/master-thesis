import numpy as np


def system_simulate(state):
    """
    TODO: This script is very similar to the scenario simulation script with the markov chain. Reuse logic
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

        # loop to generate trips from the cluster
        for j in range(number_of_trips):
            end_cluster = np.random.choice(
                sorted(state.clusters, key=lambda state_cluster: state_cluster.id),
                p=start_cluster.get_leave_distribution(),
            )

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
