import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from classes.Cluster import Cluster
from classes.Scooter import Scooter


def read_bounded_csv_file(
    file_path: str, boundary: tuple, separator=";"
) -> pd.DataFrame:
    """
    Reads csv file from Entur and outputs a dataframe
    with scooters within the given boundary
    :param file_path: filepath to csv file
    :param boundary: format: (lat, lon, lat, lon)
    :param separator: how to separate the values in a row of the csv default ";"
    :return: dataframe with scooter data
    """
    # Get EnTur data from csv file
    df = pd.read_csv(file_path, sep=separator)
    # Hardcoded boundary on data
    lat_min, lat_max, lon_min, lon_max = boundary
    # Filter out data not within boundary
    return df.loc[
        (
            (lon_min <= df["lon"])
            & (df["lon"] <= lon_max)
            & (lat_min <= df["lat"])
            & (df["lat"] <= lat_max)
        )
    ]


def cluster_data(data: pd.DataFrame) -> [int]:
    """
    Uses an clustering algorithm to group togeter scooters
    :param data: geospatial data containing cols ["lat", "lon"]
    :return: list of labels for input data
    """
    # Generate numpy array from dataframe
    coords = data[["lat", "lon"]].values
    # Run k-means algorithm to generate clusters
    return KMeans().fit(coords).labels_


def generate_cluster_objects(
    scooter_data: pd.DataFrame, cluster_labels: list
) -> [Cluster]:
    """
    Based on cluster labels and scooter data create Scooter and Cluster objects.
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
        clusters.append(Cluster(scooters))
    return clusters


def plot_cluster_data(data: pd.DataFrame, cluster_centers: pd.DataFrame):
    fig, ax = plt.subplots(figsize=[10, 6])
    rs_scatter = ax.scatter(
        cluster_centers["lon"],
        cluster_centers["lat"],
        c="#99cc99",
        edgecolor="None",
        alpha=0.7,
        s=120,
    )
    df_scatter = ax.scatter(data["lon"], data["lat"], c="k", alpha=0.1, s=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(
        [df_scatter, rs_scatter], ["Full dataset", "Cluster centers"], loc="upper right"
    )
    plt.show()
