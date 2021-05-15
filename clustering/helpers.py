import copy
import math

import numpy as np
import pandas as pd
from scipy import stats
import globals
import system_simulation.scripts


def normalize_to_integers(array, sum_to=1):
    normalized_cluster_ideal_states = sum_to * array / sum(array)
    rests = normalized_cluster_ideal_states - np.floor(normalized_cluster_ideal_states)
    number_of_ones = round(sum(rests))
    sorted_rests = np.sort(rests)
    return np.array(
        np.floor(normalized_cluster_ideal_states)
        + [1 if rest in sorted_rests[-number_of_ones:] else 0 for rest in rests],
        dtype="int32",
    ).tolist()


def scooter_sample_filter(dataframe: pd.DataFrame, sample_size=None):
    if sample_size:
        dataframe = dataframe.sample(sample_size)
    return dataframe["id"].tolist()


def merge_scooter_snapshots(state, first_snapshot_data, second_snapshot_data):
    # Join tables on scooter id
    merged_tables = pd.merge(
        left=first_snapshot_data,
        right=second_snapshot_data,
        left_on="id",
        right_on="id",
        how="inner",
        suffixes=("_before", "_after"),
    )

    # Filtering out scooters that has moved during the 20 minutes
    moved_scooters = get_moved_scooters(merged_tables)
    moved_scooters["cluster_before"] = [
        state.get_cluster_by_lat_lon(row["lat_before"], row["lon_before"]).id
        for index, row in moved_scooters.iterrows()
    ]
    moved_scooters["cluster_after"] = [
        state.get_cluster_by_lat_lon(row["lat_after"], row["lon_after"]).id
        for index, row in moved_scooters.iterrows()
    ]
    # Remove the scooters who move, but within the same cluster
    filtered_moved_scooters = moved_scooters[
        moved_scooters["cluster_before"] != moved_scooters["cluster_after"]
    ].copy()

    # Due to the dataset only showing available scooters we need to find out how many scooters leave the zone
    # resulting in a battery percent below 20. To find these scooters we find scooters from the first snapshot
    # that is not in the merge. The "~" symbol indicates "not" in pandas boolean indexing
    disappeared_scooters: pd.DataFrame = first_snapshot_data.loc[
        ~first_snapshot_data["id"].isin(merged_tables["id"])
    ].copy()
    # Find the origin cluster for these scooters
    disappeared_scooters["cluster_id"] = [
        state.get_cluster_by_lat_lon(row["lat"], row["lon"])
        for index, row in disappeared_scooters.iterrows()
    ]
    return moved_scooters, filtered_moved_scooters, disappeared_scooters


def get_moved_scooters(merged_tables):
    return merged_tables[
        (merged_tables["battery_before"] > merged_tables["battery_after"])
        & (abs(merged_tables["lat_before"] - merged_tables["lat_after"]) >= 0.0001)
        & (abs(merged_tables["lon_before"] - merged_tables["lon_after"]) >= 0.0001)
    ].copy()


def idealize_state(state):
    state_rebalanced_ideal_state = copy.deepcopy(state)

    # Set all clusters to ideal state
    excess_scooters = []
    for cluster in state_rebalanced_ideal_state.clusters:
        # Swap all scooters under 50% battery
        for scooter in cluster.scooters:
            if scooter.battery < 50:
                scooter.swap_battery()
        # Find scooters possible to pick up
        positive_deviation = len(cluster.get_available_scooters()) - cluster.ideal_state
        if positive_deviation > 0:
            # Add scooters possible to pick up
            excess_scooters += [
                (scooter, cluster) for scooter in cluster.scooters[:positive_deviation]
            ]

    # Add excess scooters to clusters in need of scooters
    for cluster in state_rebalanced_ideal_state.clusters:
        # Find out how many scooters to add to cluster
        number_of_scooters_to_add = cluster.ideal_state - len(
            cluster.get_available_scooters()
        )
        # Add scooters to the cluster only if the number of available scooter is lower than ideal state
        if number_of_scooters_to_add > 0:
            for _ in range(number_of_scooters_to_add):
                if len(excess_scooters) > 0:
                    # fetch and remove a scooter from the excess scooters
                    scooter, origin_cluster = excess_scooters.pop()
                    # Remove scooter from old cluster
                    origin_cluster.remove_scooter(scooter)
                    # Add scooter to new cluster
                    cluster.add_scooter(scooter)

    return state_rebalanced_ideal_state


def simulate_state_outcomes(state_rebalanced_ideal_state, state):
    # dict to record the outcomes of available scooters in a cluster after simulation
    simulating_outcomes = {
        cluster_id: []
        for cluster_id in range(len(state_rebalanced_ideal_state.clusters))
    }

    # simulating 100 times
    for i in range(100):
        simulating_state = copy.deepcopy(state_rebalanced_ideal_state)
        # simulates until the end of the day
        for j in range(
            round(
                globals.HyperParameters().SHIFT_DURATION
                / globals.ITERATION_LENGTH_MINUTES
            )
        ):
            system_simulation.scripts.system_simulate(simulating_state)

        # recording the available scooters in every cluster after a day
        for cluster in simulating_state.clusters:
            simulating_outcomes[cluster.id].append(
                len(cluster.get_available_scooters())
            )

    new_ideal_states = {}

    delta_ideal_state_and_outcomes = {}
    for cluster in state_rebalanced_ideal_state.clusters:
        simulating_outcome = simulating_outcomes[cluster.id]
        # setting the new ideal state to trip intensity if min of all outcomes is larger than ideal state
        # -> in all scenarios there is a positive inflow to the cluster
        if min(simulating_outcome) > cluster.ideal_state:
            new_ideal_states[cluster.id] = math.ceil(
                cluster.trip_intensity_per_iteration
            )
        # calculating the difference from outcomes and ideal state
        else:
            delta_ideal_state_and_outcomes[cluster.id] = [
                cluster.ideal_state - outcome
                for outcome in simulating_outcomes[cluster.id]
            ]

    # initial parameter for the percentile and delta of the percentile
    percentile = 1.0
    delta = 0.01

    # loop until he sum of new ideal states is less or equal to the number of scooters in the state
    while True:
        # setting the new ideal state of the clusters where not all outcomes are greater than ideal state to:
        # ideal state + %-percentile of the difference between ideal state - outcomes
        for cluster_id in delta_ideal_state_and_outcomes.keys():
            quantile_outcomes = round(
                np.quantile(delta_ideal_state_and_outcomes[cluster_id], percentile)
            )

            new_ideal_states[cluster_id] = (
                state_rebalanced_ideal_state.clusters[cluster_id].ideal_state
                + quantile_outcomes
                if quantile_outcomes > 0
                else 0
            )

        sum_ideal_state = sum(list(new_ideal_states.values()))

        # breaking
        if sum_ideal_state <= len(state.get_scooters()):
            for cluster_id in new_ideal_states.keys():
                state.clusters[cluster_id].average_number_of_scooters = state.clusters[
                    cluster_id
                ].ideal_state
                state.clusters[cluster_id].ideal_state = new_ideal_states[cluster_id]
            break
        elif percentile <= 0:
            raise ValueError(
                "0%-percentile of ideal state-outcomes are greater than the number of scooters"
            )
        else:
            percentile -= delta
