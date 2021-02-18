import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from classes import Cluster, Scooter
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
    raw_data = pd.read_csv(file_path, sep=separator).set_index("id")
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
    scooter_data: pd.DataFrame, cluster_labels: list
) -> np.ndarray:
    """
    Based on the clusters created, based on the scooter_data i.e. cluster_labels, a matrix corresponding to the
    probability that a scooter will move between two clusters.
    E.g. probability_matrix[3][5] - will return the probability for a scooter of moving from cluster 3 to 5
    probability_matrix[3][3] - will return the probability of a scooter staying in cluster 3
    :param scooter_data: geospatial data for scooters
    :param cluster_labels: list of labels for scooter data
    :return: probability matrix
    """
    number_of_clusters = len(np.unique(cluster_labels))
    delayed_data = read_bounded_csv_file(
        "test_data/0920-entur-snapshot.csv", GEOSPATIAL_BOUND_NEW, separator=","
    )

    return np.random.rand(number_of_clusters, number_of_clusters)


def generate_cluster_objects(
    scooter_data: pd.DataFrame, cluster_labels: list, probability_matrix: np.ndarray
) -> [Cluster]:
    """
    Based on cluster labels and scooter data create Scooter and Cluster objects.
    :param scooter_data: geospatial data for scooters
    :param cluster_labels: list of labels for scooter data
    :param probability_matrix: see scooter_movement_analysis function for explanation
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
        clusters.append(
            Cluster(cluster_label, scooters, probability_matrix[cluster_label])
        )
    return clusters
