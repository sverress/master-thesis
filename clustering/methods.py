import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from classes import State, Scooter, Cluster
from globals import GEOSPATIAL_BOUND_NEW


def read_bounded_csv_file(
    file_path: str, boundary: tuple, sample_size=None, separator=";", operator=None
) -> pd.DataFrame:
    """
    Reads csv file from Entur and outputs a dataframe
    with scooters within the given boundary
    :param file_path: filepath to csv file
    :param boundary: format: (lat, lon, lat, lon)
    :param sample_size: integer number with number of scooters to fetch
    :param separator: how to separate the values in a row of the csv default ";"
    :param operator: Either "voi" or "tier"'
    :return: dataframe with scooter data
    """
    # Get EnTur data from csv file
    raw_data = pd.read_csv(file_path, sep=separator)
    # Hardcoded boundary on data
    lat_min, lat_max, lon_min, lon_max = boundary
    # Filter out data not within boundary
    raw_data = raw_data.loc[
        (
            (lon_min <= raw_data["lon"])
            & (raw_data["lon"] <= lon_max)
            & (lat_min <= raw_data["lat"])
            & (raw_data["lat"] <= lat_max)
        )
    ]

    # Only two operators in the current dataset
    if operator == "voi" or operator == "tier":
        # Reduce number of scooters with a operator filter
        raw_data = raw_data[raw_data["operator"] == operator]

    if sample_size:
        # Reduce number of scooters with a sample size
        raw_data = raw_data.sample(sample_size)

    return raw_data


def cluster_data(data: pd.DataFrame, number_of_clusters: int) -> [int]:
    """
    Uses an clustering algorithm to group together scooters
    :param data: geospatial data containing cols ["lat", "lon"]
    :param number_of_clusters: how many clusters to create
    :return: list of labels for input data
    """
    # Generate numpy array from dataframe
    coords = data[["lat", "lon"]].values
    # Run k-means algorithm to generate clusters
    return KMeans(number_of_clusters).fit(coords).labels_


def scooter_movement_analysis(
    state: State, scooter_data: pd.DataFrame, cluster_labels: list
) -> np.ndarray:
    """
    Based on the clusters created, based on the scooter_data i.e. cluster_labels, a matrix corresponding to the
    probability that a scooter will move between two clusters.
    E.g. probability_matrix[3][5] - will return the probability for a scooter of moving from cluster 3 to 5
    probability_matrix[3][3] - will return the probability of a scooter staying in cluster 3
    :param state: list of generated clusters
    :param scooter_data: geospatial data for scooters
    :param cluster_labels: list of labels for scooter data
    :return: probability matrix
    """

    # Adding cluster labels to scooter data
    scooter_data["cluster_id"] = cluster_labels

    # Fetching snapshot 20 minutes after original snapshot
    delayed_data = read_bounded_csv_file(
        "test_data/0920-entur-snapshot.csv", GEOSPATIAL_BOUND_NEW, separator=","
    )

    # Join tables on scooter id
    merged_tables = pd.merge(
        left=scooter_data, right=delayed_data, left_on="id", right_on="id", how="inner"
    )

    # Filtering out scooters that has moved during the 20 minutes
    moved_scooters = merged_tables[
        merged_tables["battery_x"] != merged_tables["battery_y"]
    ]

    # Due to the dataset only showing available scooters we need to find out how many scooters leave the zone resulting
    # in a battery percent below 20. To find these scooters we find scooters from the first snapshot that is not in the
    # merge. The "~" symbol indicates "not" in pandas boolean indexing
    disappeared_scooters = scooter_data[~scooter_data["id"].isin(merged_tables["id"])]

    # Initialize probability_matrix with number of scooters in each cluster
    number_of_clusters = len(np.unique(cluster_labels))
    number_of_scooters = np.array(
        [[cluster.number_of_scooters() for cluster in state.clusters]]
        * number_of_clusters,
        dtype="float64",
    ).transpose()

    # Create counter for every combination of cluster and count how many scooters move
    move_count = np.zeros((number_of_clusters, number_of_clusters), dtype="float64")
    for index, row in moved_scooters.iterrows():
        # Find the nearest cluster the scooter now belongs to
        new_cluster = state.get_cluster_by_lat_lon(row["lat_y"], row["lon_y"])
        # Increase the counter for every visit
        move_count[row["cluster_id"]][new_cluster.id] += 1

    # Calculate number of scooters who stayed in each zone
    for cluster_id in np.unique(cluster_labels):
        # Formula: Number of scooters from beginning - number of scooters leaving - # disappeared scooters
        move_count[cluster_id][cluster_id] = (
            number_of_scooters[cluster_id][cluster_id]
            - sum(move_count[cluster_id][np.arange(number_of_clusters) != cluster_id])
            # np.arange(number_of_clusters) != cluster_id => all indices except cluster_id
            - len(
                disappeared_scooters[disappeared_scooters["cluster_id"] == cluster_id]
            )
        )

    probability_matrix = move_count / number_of_scooters

    # Normalize non stay distribution - Same as distribute
    for cluster_id in np.unique(cluster_labels):
        # Calculate probability of a scooter leaving the cluster
        break
    return probability_matrix


def generate_cluster_objects(
    scooter_data: pd.DataFrame, cluster_labels: list
) -> [Cluster]:
    """
    Based on cluster labels and scooter data create Scooter and Cluster objects.
    Cluster class generates cluster center
    :param scooter_data: geospatial data for scooters
    :param cluster_labels: list of labels for scooter data
    :return: list of clusters
    """
    # Generate series of scooters belonging to each cluster
    clusters = []
    for cluster_label in np.unique(cluster_labels):
        # Filter out scooters within cluster
        cluster_scooters = scooter_data[cluster_labels == cluster_label]
        # Generate scooter objets, using index as ID
        scooters = [
            Scooter(row["lat"], row["lon"], row["battery"], index)
            for index, row in cluster_scooters.iterrows()
        ]
        clusters.append(Cluster(cluster_label, scooters))
    return clusters
