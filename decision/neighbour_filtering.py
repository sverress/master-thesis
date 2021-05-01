import numpy as np
import helpers
from globals import DEFAULT_NUMBER_OF_NEIGHBOURS, DEFAULT_NUMBER_OF_RANDOM_NEIGHBOURS


def filtering_neighbours(
    state,
    vehicle,
    number_of_neighbours=DEFAULT_NUMBER_OF_NEIGHBOURS,
    number_of_random_neighbours=DEFAULT_NUMBER_OF_RANDOM_NEIGHBOURS,
    time=None,
    exclude=None,
    max_swaps=0,
):
    """
    Filtering out neighbours based on a score of deviation of ideal state and distance from current cluster
    :param max_swaps: max possible battery swaps of all actions
    :param exclude: locations to be excluded as neighbours
    :param time: time of the world so charging at depot can be controlled
    :param vehicle: vehicle to find neighbours from
    :param state: state object to evaluate
    :param number_of_neighbours: number of neighbours to be returned
    :param number_of_random_neighbours: int - number of random neighbours to be added to the neighbourhood list
    :return:
    """
    exclude = exclude if exclude else []
    distance_to_all_clusters = state.get_distance_to_all_clusters(
        vehicle.current_location.id
    )
    distance_scores = helpers.normalize_list(distance_to_all_clusters)

    if len(vehicle.scooter_inventory) > 0:
        # If the vehicle has scooters in its inventory, we want to value clusters far away from the ideal state
        cluster_values = get_deviation_ideal_state(state)
    else:
        # If the vehicle has no scooters in its inventory,
        # it is more interesting to look at clusters with deficient batteries
        cluster_values = get_battery_deficient_in_clusters(state)
    # Low values are desirable. Hence, a high deviation or battery deficient should give a low value
    cluster_values = [1 - value for value in helpers.normalize_list(cluster_values)]

    # Sort clusters by the sum of distance and score. Exclude current cluster and excluded clusters
    all_sorted_neighbours = sorted(
        [
            cluster
            for cluster in state.clusters
            if cluster.id != vehicle.current_location.id and cluster.id not in exclude
        ],
        key=lambda cluster: distance_scores[cluster.id] + cluster_values[cluster.id],
    )

    # Reduce number of neighbours to "number_of_neighbours" and add random neighbours
    neighbours = (
        all_sorted_neighbours[: number_of_neighbours - number_of_random_neighbours]
        + np.random.choice(
            all_sorted_neighbours[number_of_neighbours - number_of_random_neighbours :],
            size=number_of_random_neighbours,
        ).tolist()
    )

    # Add depot neighbours and return
    return neighbours + add_depots_as_neighbours(state, time, vehicle, max_swaps)


def add_depots_as_neighbours(state, time, vehicle, max_swaps):
    """
    Adds big depot and closest available (able to change all flat batteries) small depot
    """
    if vehicle.is_at_depot() or not (
        time
        and vehicle.battery_inventory - max_swaps
        < vehicle.battery_inventory_capacity * 0.2
    ):
        return []

    big_depot, *small_depots = state.depots
    # Filter out small depots that are not able to change all flat batteries for current vehicles
    available_small_depots = [
        depot
        for depot in small_depots
        if depot.get_available_battery_swaps(time) >= vehicle.flat_batteries()
    ]
    return (
        [big_depot]
        + [
            min(
                [depot for depot in available_small_depots],
                key=lambda depot: state.get_distance(
                    vehicle.current_location.id, depot.id
                ),
            )
        ]
        if available_small_depots
        else [big_depot]
    )


def get_deviation_ideal_state(state):
    # cluster score based on deviation
    return [cluster.ideal_state - len(cluster.scooters) for cluster in state.clusters]


def get_battery_deficient_in_clusters(state):
    # cluster score based on how much deficient of battery the cluster have
    return [
        len(cluster.scooters) - cluster.get_current_state()
        for cluster in state.clusters
    ]
