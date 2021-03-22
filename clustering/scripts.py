from classes import Vehicle, State
import clustering.methods as methods
import os
from globals import STATE_CACHE_DIR
from progress.bar import Bar


def get_initial_state(
    sample_size=None, number_of_clusters=20, save=True, cache=True
) -> State:
    # If this combination has been requested before we fetch a cached version
    if cache and os.path.exists(
        f"{STATE_CACHE_DIR}/c{number_of_clusters}s{sample_size}.pickle"
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

    clustering = Bar("| Clustering data", max=4)
    # Get dataframe from EnTur CSV file within boundary
    entur_dataframe = methods.read_bounded_csv_file("test_data/0900-entur-snapshot.csv")
    clustering.next()

    # Create clusters
    cluster_labels = methods.cluster_data(entur_dataframe, number_of_clusters)
    clustering.next()

    # Structure data into objects
    clusters = methods.generate_cluster_objects(entur_dataframe, cluster_labels)
    clustering.next()

    # generate depots and adding them to clusters list
    depots = methods.generate_depots(number_of_clusters=len(clusters))
    clustering.next()

    # Choosing first cluster as starting cluster in state
    current_location = depots[0]
    clustering.finish()

    # Choosing a default vehicle as the vehicle in the new state
    vehicle = Vehicle()

    # Create state object
    initial_state = State(clusters, depots, current_location, vehicle)

    # Sample size filtering. Create list of scooter ids to include
    sample_scooters = methods.scooter_sample_filter(entur_dataframe, sample_size)

    # Find the ideal state for each cluster
    initial_state.compute_and_set_ideal_state(sample_scooters)

    # Trip intensity analysis
    initial_state.compute_and_set_trip_intensity(sample_scooters)

    # Get probability of movement from scooters in a cluster
    probability_matrix = methods.scooter_movement_analysis(initial_state)
    initial_state.set_probability_matrix(probability_matrix)

    if sample_size:
        initial_state.sample(sample_size)

    if save:
        # Cache the state for later
        initial_state.save_state()
        print("Setup state completed\n")

    return initial_state


if __name__ == "__main__":
    get_initial_state(sample_size=100, number_of_clusters=2)
