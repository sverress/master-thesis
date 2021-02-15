from classes.State import State
import numpy as np


def system_simulate(state: State):
    """
    Simulation of poisson process on the system ->
    Poisson distributed number of trips, markov chain decides where the trip goes
    :param state: current state
    :return: new state after a simulation of the poisson process
    """
    min_battery = 20.0
    trips = []
    for i, cluster in enumerate(state.clusters):
        # poisson process to select number of trips in a iteration
        number_of_trips = round(np.random.poisson(cluster.trip_intensity_per_iteration))

        # can't complete more trips then there is scooters with battery over min_battery
        valid_scooters = sum(
            [1 for scooter in cluster.scooters if scooter.battery >= min_battery]
        )
        if number_of_trips > valid_scooters:
            number_of_trips = valid_scooters

        start_cluster = cluster
        # loop to generate trips from the cluster
        for j in range(number_of_trips):
            # with this implementation, a trip can be within a cluster
            end_cluster = round(np.random.uniform(0, len(state.clusters) - 1))
            trips.append(
                (start_cluster, state.clusters[end_cluster], start_cluster.scooters[j])
            )

    # compute trip after all trips are generated to avoid handling inflow in cluster
    for cluster_trips in trips:
        start_cluster, end_cluster, scooter = cluster_trips
        start_cluster.scooters.remove(scooter)
        scooter.lat = end_cluster.center[0]
        scooter.lon = end_cluster.center[1]
        trip_distance = state.get_distance(start_cluster, end_cluster)
        scooter.change_battery(trip_distance)
        end_cluster.add_scooter(scooter)

    return state
