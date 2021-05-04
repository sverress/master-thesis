from classes import Vehicle, State
import clustering.methods as methods
import os
from globals import *


def get_initial_state(
    sample_size=None,
    number_of_clusters=20,
    save=True,
    cache=True,
    initial_location_depot=True,
    number_of_vans=NUMBER_OF_VANS,
    number_of_bikes=NUMBER_OF_BIKES,
    ideal_state_computation=False,
) -> State:
    # If this combination has been requested before we fetch a cached version
    filepath = (
        f"{STATE_CACHE_DIR}/"
        f"{State.save_path(number_of_clusters, sample_size, ideal_state_computation)}.pickle"
    )
    if cache and os.path.exists(filepath):
        print(f"\nUsing cached version of state from {filepath}\n")
        initial_state = State.load(filepath)
    else:

        print(
            f"\nSetup initial state from entur dataset with {number_of_clusters} clusters and {sample_size} scooters"
        )
        # Get dataframe from EnTur CSV file within boundary
        entur_dataframe = methods.read_bounded_csv_file(
            "test_data/0900-entur-snapshot.csv"
        )

        # Create clusters
        cluster_labels = methods.cluster_data(entur_dataframe, number_of_clusters)

        # Structure data into objects
        clusters = methods.generate_cluster_objects(entur_dataframe, cluster_labels)

        # generate depots and adding them to clusters list
        depots = methods.generate_depots(number_of_clusters=len(clusters))

        # Create state object
        initial_state = State(clusters, depots, [])

        # Sample size filtering. Create list of scooter ids to include
        sample_scooters = methods.scooter_sample_filter(entur_dataframe, sample_size)

        # Find the ideal state for each cluster
        initial_state.compute_and_set_ideal_state(sample_scooters)

        # Trip intensity analysis
        initial_state.compute_and_set_trip_intensity(
            sample_scooters, ideal_state_computation
        )

        # Get probability of movement from scooters in a cluster
        probability_matrix = methods.scooter_movement_analysis(initial_state)
        initial_state.set_probability_matrix(probability_matrix)

        if sample_size:
            initial_state.sample(sample_size)

        # Generate scenarios
        initial_state.simulation_scenarios = methods.generate_scenarios(initial_state)

        if save:
            # Cache the state for later
            initial_state.save_state()
            print("Setup state completed\n")

    # Choosing a location as starting cluster for all vehicles
    current_location = (
        initial_state.depots[0] if initial_location_depot else initial_state.clusters[0]
    )

    # Setting vehicles to initial state
    initial_state.vehicles = [
        Vehicle(i, current_location, VAN_BATTERY_INVENTORY, VAN_SCOOTER_INVENTORY)
        for i in range(number_of_vans)
    ] + [
        Vehicle(i, current_location, BIKE_BATTERY_INVENTORY, BIKE_SCOOTER_INVENTORY)
        for i in range(number_of_vans, number_of_vans + number_of_bikes)
    ]

    return initial_state


if __name__ == "__main__":
    get_initial_state(sample_size=100, number_of_clusters=2)
