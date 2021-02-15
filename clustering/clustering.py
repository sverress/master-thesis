import pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from shapely.geometry import MultiPoint


def cluster_data() -> (pd.DataFrame, pd.DataFrame):
    def get_center_of_cluster(cluster):
        cluster_centroid = MultiPoint(cluster[["lat", "lon"]].values).centroid
        return cluster_centroid.x, cluster_centroid.y

    # Get EnTur data from csv file
    df = pd.read_csv("project_thesis/test_data/bigquery-results.csv", sep=";")
    # Hardcoded boundary on data
    lat_min, lat_max, lon_min, lon_max = (
        59.9112,
        59.9438,
        10.7027,
        10.7772,
    )
    # Filter out data not within boundary
    df = df.loc[
        (
            (lon_min <= df["lon"])
            & (df["lon"] <= lon_max)
            & (lat_min <= df["lat"])
            & (df["lat"] <= lat_max)
        )
    ]
    # Generate numpy array from dataframe
    coords = df[["lat", "lon"]].values
    # Run k-means algorithm to generate clusters
    db = KMeans().fit(coords)
    # Add cluster labels to dataframe
    cluster_labels = db.labels_
    df["cluster"] = cluster_labels
    # Generate series of scooters belonging to each cluster
    clusters = pd.Series(
        [df[cluster_labels == n] for n in range(len(set(cluster_labels)))]
    )
    # Find center for all clusters
    center_points = clusters.map(get_center_of_cluster)
    lats, lons = zip(*center_points)
    return df, pd.DataFrame({"lon": lons, "lat": lats})


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


if __name__ == "__main__":
    plot_cluster_data(*cluster_data())
