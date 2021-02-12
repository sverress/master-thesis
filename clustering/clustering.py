# https://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

df = pd.read_csv("test_data/bigquery-results.csv", sep=";")
coords = df[["lat", "lon"]].values
kms_per_radian = 6371.0088
km_within_cluster = 0.100
epsilon = km_within_cluster / kms_per_radian
db = DBSCAN(eps=epsilon, min_samples=1, algorithm="ball_tree", metric="haversine").fit(
    np.radians(coords)
)
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print("Number of clusters: {}".format(num_clusters))
