from globals import BATTERY_INVENTORY, MAX_DISTANCE
import bisect
import numpy as np


def filtering_neighbours(
    state,
    vehicle,
    number_of_neighbours=3,
    number_of_random_neighbours=0,
    time=None,
    exclude=None,
):
    """
    Filtering out neighbours based on a score of deviation of ideal state and distance from current cluster
    :param exclude: locations to be excluded as neighbours
    :param time: time of the world so charging at depot can be controlled
    :param vehicle: vehicle to find neighbours from
    :param state: state object to evaluate
    :param number_of_neighbours: number of neighbours to be returned
    :param number_of_random_neighbours: int - number of random neighbours to be added to the neighbourhood list
    :return:
    """
    exclude = exclude if exclude else []
    clusters = state.clusters
    distance_to_all_clusters = state.get_distance_to_all_clusters(
        vehicle.current_location.id
    )
    max_dist, min_dist = max(distance_to_all_clusters), min(distance_to_all_clusters)
    distance_scores = [
        (dist - min_dist) / (max_dist - min_dist) for dist in distance_to_all_clusters
    ]
    if len(vehicle.scooter_inventory) > 0:
        cluster_value = get_deviation_ideal_state(state)
    else:
        cluster_value = get_battery_deficient_in_clusters(state)

    max_cluster_value, min_cluster_value = (
        max(cluster_value),
        min(cluster_value),
    )

    if max_cluster_value == min_cluster_value:
        cluster_score = [1] * len(cluster_value)
    else:
        cluster_score = [
            1
            - (
                (deviation - min_cluster_value)
                / (max_cluster_value - min_cluster_value)
            )
            for deviation in cluster_value
        ]

    score_indices = []
    total_score_list = []
    for state_cluster in clusters:
        cluster_id = state_cluster.id
        if cluster_id != vehicle.current_location.id and cluster_id not in exclude:
            total_score = distance_scores[cluster_id] + cluster_score[cluster_id]
            index = bisect.bisect(total_score_list, total_score)
            total_score_list.insert(index, total_score)
            score_indices.insert(index, cluster_id)

    if number_of_random_neighbours > 0:
        neighbours = [
            clusters[index]
            for index in score_indices[
                : number_of_neighbours - number_of_random_neighbours
            ]
            + np.random.choice(
                score_indices[number_of_neighbours - number_of_random_neighbours :],
                size=number_of_random_neighbours,
            ).tolist()
        ]

    else:
        neighbours = [clusters[index] for index in score_indices[:number_of_neighbours]]

    return (
        neighbours + add_depots_as_neighbours(state, time, vehicle)
        if time and vehicle.battery_inventory < BATTERY_INVENTORY * 0.2
        else neighbours
    )


def add_depots_as_neighbours(state, time, vehicle):
    depots = []
    if vehicle.is_at_depot():
        return depots
    else:
        closest_small_depot = None
        closest_distance = MAX_DISTANCE
        for i, depot in enumerate(state.depots):
            if i == 0:
                depots.append(depot)
            else:
                distance_to_depot = state.get_distance_locations(
                    vehicle.current_location, depot
                )
                if (
                    closest_distance > distance_to_depot
                    and depot.get_available_battery_swaps(time)
                    >= BATTERY_INVENTORY - vehicle.battery_inventory
                ):
                    closest_small_depot = depot
                    closest_distance = distance_to_depot

        if closest_small_depot:
            depots.append(closest_small_depot)

        return depots


def get_deviation_ideal_state(state):
    # cluster score based on deviation
    return [
        abs(cluster.ideal_state - len(cluster.scooters)) for cluster in state.clusters
    ]


def get_battery_deficient_in_clusters(state):
    # cluster score based on how much deficient of battery the cluster have
    return [
        len(cluster.scooters) - cluster.get_current_state()
        for cluster in state.clusters
    ]
