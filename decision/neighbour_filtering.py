from globals import BATTERY_LIMIT
import bisect
import numpy as np


def filtering_neighbours(state, number_of_neighbours=3, random_neighbour=False):
    clusters = state.clusters
    distance_to_all_clusters = state.get_distance_to_all(state.current_cluster.id)
    max_dist, min_dist = max(distance_to_all_clusters), min(distance_to_all_clusters)
    distance_scores = [
        (dist - min_dist) / (max_dist - min_dist) for dist in distance_to_all_clusters
    ]

    deviation_ideal_states = [
        abs(
            cluster.ideal_state
            - sum(
                [
                    scooter.battery
                    for scooter in cluster.get_valid_scooters(BATTERY_LIMIT)
                ]
            )
            / 100
        )
        for cluster in clusters
    ]

    max_deviation, min_deviation = (
        max(deviation_ideal_states),
        min(deviation_ideal_states),
    )

    deviation_scores = [
        1 - ((deviation - min_deviation) / (max_deviation - min_deviation))
        for deviation in deviation_ideal_states
    ]

    score_indices = []
    total_score_list = []
    for cluster in clusters:
        cluster_id = cluster.id
        if cluster_id != state.current_cluster.id:
            total_score = distance_scores[cluster_id] + deviation_scores[cluster_id]
            index = bisect.bisect(total_score_list, total_score)
            total_score_list.insert(index, total_score)
            score_indices.insert(index, cluster_id)

    return (
        [clusters[index] for index in score_indices[: number_of_neighbours - 1]]
        + [clusters[np.random.choice(score_indices[number_of_neighbours - 1 :])]]
        if random_neighbour
        else [clusters[index] for index in score_indices[:number_of_neighbours]]
    )
