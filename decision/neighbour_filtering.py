from globals import BATTERY_LIMIT, BATTERY_INVENTORY, MAX_DISTANCE
import classes
import bisect
import numpy as np


def filtering_neighbours(state, number_of_neighbours=3, random_neighbours=0, time=-1):
    """
    Filtering out neighbours based on a score of deviation of ideal state and distance from current cluster
    :param state: state object to evaluate
    :param number_of_neighbours: number of neighbours to be returned
    :param random_neighbours: int - number of random neighbours to be added to the neighbourhood list
    :return:
    """
    clusters = state.clusters
    distance_to_all_clusters = state.get_distance_to_all_clusters(
        state.current_location.id
    )
    max_dist, min_dist = max(distance_to_all_clusters), min(distance_to_all_clusters)
    distance_scores = [
        (dist - min_dist) / (max_dist - min_dist) for dist in distance_to_all_clusters
    ]

    deviation_ideal_states = [
        abs(cluster.ideal_state - len(cluster.get_valid_scooters(BATTERY_LIMIT)))
        for cluster in clusters
    ]

    max_deviation, min_deviation = (
        max(deviation_ideal_states),
        min(deviation_ideal_states),
    )

    if max_deviation == min_deviation:
        deviation_scores = [1] * len(deviation_ideal_states)
    else:
        deviation_scores = [
            1 - ((deviation - min_deviation) / (max_deviation - min_deviation))
            for deviation in deviation_ideal_states
        ]

    score_indices = []
    total_score_list = []
    for cluster in clusters:
        cluster_id = cluster.id
        if cluster_id != state.current_location.id:
            total_score = distance_scores[cluster_id] + deviation_scores[cluster_id]
            index = bisect.bisect(total_score_list, total_score)
            total_score_list.insert(index, total_score)
            score_indices.insert(index, cluster_id)

    if random_neighbours > 0:
        neighbours = [
            clusters[index]
            for index in score_indices[: number_of_neighbours - random_neighbours]
        ] + [
            clusters[
                np.random.choice(
                    score_indices[number_of_neighbours - random_neighbours :]
                )
            ]
        ]
    else:
        neighbours = [clusters[index] for index in score_indices[:number_of_neighbours]]

    return (
        neighbours + add_depots_as_neighbours(state, time) if time > -1 else neighbours
    )


def add_depots_as_neighbours(state, time):
    depots = []
    if isinstance(state.current_location, classes.Depot):
        return depots
    else:
        closest_small_depot = None
        closest_distance = MAX_DISTANCE
        for i, depot in enumerate(state.depots):
            if i == 0:
                depots.append(depot)
            else:
                distance_to_depot = state.get_distance_locations(
                    state.current_location, depot
                )
                if (
                    closest_distance > distance_to_depot
                    and depot.get_available_battery_swaps(time)
                    >= BATTERY_INVENTORY - state.vehicle.battery_inventory
                ):
                    closest_small_depot = depot
                    closest_distance = distance_to_depot

        if closest_small_depot:
            depots.append(closest_small_depot)

        return depots
