from classes import State, Vehicle
from clustering.methods import (
    read_bounded_csv_file,
    cluster_data,
    generate_cluster_objects,
    scooter_movement_analysis,
)
from globals import GEOSPATIAL_BOUND_NEW


def get_initial_state(sample_size=None, number_of_clusters=20) -> State:

    # Get dataframe from EnTur CSV file within boundary
    entur_dataframe = read_bounded_csv_file(
        "test_data/0900-entur-snapshot.csv",
        GEOSPATIAL_BOUND_NEW,
        sample_size=sample_size,
    )

    # Create clusters
    cluster_labels = cluster_data(entur_dataframe, number_of_clusters)

    # Structure data into objects
    clusters = generate_cluster_objects(entur_dataframe, cluster_labels)

    # Choosing first cluster as starting cluster in state
    current_cluster = clusters[0]

    # Choosing a default vehicle as the vehicle in the new state
    vehicle = Vehicle()

    initial_state = State(clusters, current_cluster, vehicle)

    # Get probability of movement from scooters in a cluster
    probability_matrix = scooter_movement_analysis(
        initial_state,
        entur_dataframe,
        read_bounded_csv_file(
            "test_data/0920-entur-snapshot.csv",
            GEOSPATIAL_BOUND_NEW,
            sample_size=sample_size,
            separator=",",
        ),
    )

    initial_state.set_probability_matrix(probability_matrix)

    return initial_state


if __name__ == "__main__":
    get_initial_state()
