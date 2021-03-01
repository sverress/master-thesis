from classes import State, Vehicle
from clustering.methods import (
    read_bounded_csv_file,
    cluster_data,
    generate_cluster_objects,
    scooter_movement_analysis,
)
import os
from globals import STATE_CACHE_DIR


def get_initial_state(sample_size=None, number_of_clusters=20) -> State:

    # If this combination has been requested before we fetch a cached version
    if os.path.exists(f"{STATE_CACHE_DIR}/c{number_of_clusters}s{sample_size}.pickle"):
        return State.load_state(
            f"{STATE_CACHE_DIR}/c{number_of_clusters}s{sample_size}.pickle"
        )

    # Get dataframe from EnTur CSV file within boundary
    entur_dataframe = read_bounded_csv_file("test_data/0900-entur-snapshot.csv")

    # Create clusters
    cluster_labels = cluster_data(entur_dataframe, number_of_clusters)

    # Structure data into objects
    clusters = generate_cluster_objects(
        entur_dataframe, cluster_labels, sample_size=sample_size
    )

    # Choosing first cluster as starting cluster in state
    current_cluster = clusters[0]

    # Choosing a default vehicle as the vehicle in the new state
    vehicle = Vehicle()

    initial_state = State(clusters, current_cluster, vehicle)

    # Get probability of movement from scooters in a cluster
    probability_matrix = scooter_movement_analysis(initial_state)

    initial_state.set_probability_matrix(probability_matrix)

    # Cache the state for later
    initial_state.save_state()

    return initial_state


if __name__ == "__main__":
    get_initial_state()
