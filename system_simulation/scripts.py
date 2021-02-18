import numpy as np


def system_simulate(state):
    """
    Simulation of poisson process on the system ->
    Poisson distributed number of trips out of each cluster, markov chain decides where the trip goes
    :param state: current state
    :return: new state after a simulation of the poisson process
    """
    trip_counter = {
        (start, end): 0
        for start in np.arange(len(state.clusters))
        for end in np.arange(len(state.clusters))
        if start != end
    }
    min_battery = 20.0
    trips = []
    for i, cluster in enumerate(state.clusters):
        # poisson process to select number of trips in a iteration
        number_of_trips = round(np.random.poisson(cluster.trip_intensity_per_iteration))

        # can't complete more trips then there is scooters with battery over min_battery
        valid_scooters = cluster.get_valid_scooters(min_battery)
        if number_of_trips > len(valid_scooters):
            number_of_trips = len(valid_scooters)

        start_cluster = cluster
        # loop to generate trips from the cluster
        for j in range(number_of_trips):
            end_cluster = start_cluster.id
            while start_cluster.id == end_cluster:
                end_cluster = round(np.random.uniform(0, len(state.clusters) - 1))

            if start_cluster.id != end_cluster:
                trips.append(
                    (start_cluster, state.clusters[end_cluster], valid_scooters[j])
                )
                trip_counter[(start_cluster.id, end_cluster)] += 1

    # compute trip after all trips are generated to avoid handling inflow in cluster
    for cluster_trips in trips:
        start_cluster, end_cluster, scooter = cluster_trips
        start_cluster.scooters.remove(scooter)
        scooter.lat = end_cluster.center[0]
        scooter.lon = end_cluster.center[1]
        trip_distance = state.get_distance(start_cluster, end_cluster)
        scooter.travel(trip_distance)
        end_cluster.add_scooter(scooter)

    return [
        (start_end[0], start_end[1], trips)
        for start_end, trips in list(trip_counter.items())
    ]
