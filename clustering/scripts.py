from classes import State, Vehicle
from clustering.methods import (
    read_bounded_csv_file,
    cluster_data,
    generate_cluster_objects,
    scooter_movement_analysis,
)
import os
from globals import STATE_CACHE_DIR
from progress.bar import Bar


def get_initial_state(sample_size=None, number_of_clusters=20, save=True) -> State:
    # If this combination has been requested before we fetch a cached version
    if save and os.path.exists(
        f"{STATE_CACHE_DIR}/c{number_of_clusters}s{sample_size if sample_size else 5345}.pickle"
    ):
        print(
            f"\nUsing cached version of state from {STATE_CACHE_DIR}/c{number_of_clusters}s{sample_size}.pickle\n"
        )
        return State.load_state(
            f"{STATE_CACHE_DIR}/c{number_of_clusters}s{sample_size}.pickle"
        )
    print(
        f"\nSetup initial state from entur dataset with {number_of_clusters} clusters and {sample_size} scooters"
    )

    clustering = Bar("| Clustering data", max=3)
    # Get dataframe from EnTur CSV file within boundary
    entur_dataframe = read_bounded_csv_file("test_data/0900-entur-snapshot.csv")
    clustering.next()

    # Create clusters
    cluster_labels = cluster_data(entur_dataframe, number_of_clusters)
    clustering.next()

    # Structure data into objects
    clusters = generate_cluster_objects(
        entur_dataframe, cluster_labels, sample_size=sample_size
    )
    clustering.next()
    # Choosing first cluster as starting cluster in state
    current_cluster = clusters[0]
    clustering.finish()

    # Choosing a default vehicle as the vehicle in the new state
    vehicle = Vehicle()

    initial_state = State(clusters, current_cluster, vehicle)
    # Find the ideal state for each cluster
    initial_state.compute_and_set_ideal_state(sample_size=sample_size)
    # Get probability of movement from scooters in a cluster
    probability_matrix = scooter_movement_analysis(initial_state)

    initial_state.compute_and_set_trip_intensity(sample_size=sample_size)

    initial_state.set_probability_matrix(probability_matrix)

    # Cache the state for later
    initial_state.save_state()
    print("Setup state completed\n")

    return initial_state


if __name__ == "__main__":
    get_initial_state(sample_size=100, number_of_clusters=2)
