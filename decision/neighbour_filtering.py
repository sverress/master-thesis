from globals import BATTERY_INVENTORY, MAX_DISTANCE
import bisect
import numpy as np


def filtering_neighbours(
    state,
    number_of_neighbours=3,
    number_of_random_neighbours=0,
    vehicle_scooter_inventory=0,
    vehicle_battery_inventory=BATTERY_INVENTORY,
    time=None,
):
    """
    Filtering out neighbours based on a score of deviation of ideal state and distance from current cluster
    :param vehicle_battery_inventory: current vehicle battery inventory
    :param vehicle_scooter_inventory: current vehicle scooter inventory
    :param time: time of the world so charging at depot can be controlled
    :param state: state object to evaluate
    :param number_of_neighbours: number of neighbours to be returned
    :param number_of_random_neighbours: int - number of random neighbours to be added to the neighbourhood list
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

    cluster_value = get_cluster_value(state, vehicle_scooter_inventory)

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
    for cluster in clusters:
        cluster_id = cluster.id
        if cluster_id != state.current_location.id:
            total_score = distance_scores[cluster_id] + cluster_score[cluster_id]
            index = bisect.bisect(total_score_list, total_score)
            total_score_list.insert(index, total_score)
            score_indices.insert(index, cluster_id)

    # TODO can be removed if cluster_value function is a good solution else we use this and old implementation
    """
        if vehicle_scooter_inventory == 0:
        score_indices = list(
            filter(
                lambda score: len(clusters[score].scooters)
                > clusters[score].ideal_state,
                score_indices,
            )
        )

    """

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
        neighbours + add_depots_as_neighbours(state, time)
        if time and vehicle_scooter_inventory < BATTERY_INVENTORY * 0.2
        else neighbours
    )


def add_depots_as_neighbours(state, time):
    depots = []
    if state.is_at_depot():
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


def get_cluster_value(state, vehicle_scooter_inventory=0):

    if vehicle_scooter_inventory > 0:
        # cluster score based on deviation
        return [
            abs(cluster.ideal_state - len(cluster.get_available_scooters()))
            for cluster in state.clusters
        ]
    else:
        # cluster score based on how much deficient of battery the cluster have
        return [
            len(cluster.scooters) - cluster.get_current_state()
            for cluster in state.clusters
        ]
