import numpy as np
import pandas as pd


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
        right=first_snapshot_data,
        left=second_snapshot_data,
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
