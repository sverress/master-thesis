from classes.State import State
from classes.Vehicle import Vehicle
from clustering.helpers import (
    read_bounded_csv_file,
    cluster_data,
    generate_cluster_objects,
)
from globals import GEOSPATIAL_BOUND, GEOSPATIAL_BOUND_NEW


def get_initial_state(sample_size=None) -> State:

    # Get dataframe from EnTur CSV file within boundary
    entur_dataframe = read_bounded_csv_file(
        "test_data/bigquery-results.csv", GEOSPATIAL_BOUND_NEW, sample_size=sample_size,
    )

    # Create clusters
    cluster_labels = cluster_data(entur_dataframe)

    # Structure data into objects
    clusters = generate_cluster_objects(entur_dataframe, cluster_labels)

    # Choosing first cluster as starting cluster in state
    current_cluster = clusters[0]

    # Choosing a default vehicle as the vehicle in the new state
    vehicle = Vehicle()

    return State(clusters, current_cluster, vehicle)


if __name__ == "__main__":
    print(get_initial_state())
