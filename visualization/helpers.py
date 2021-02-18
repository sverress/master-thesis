from classes import Cluster
from globals import BLACK, GEOSPATIAL_BOUND_NEW
from itertools import cycle
import networkx as nx
import numpy as np


def make_graph(clusters: [Cluster]):
    cartesian_clusters = convert_geographic_to_cart(clusters, GEOSPATIAL_BOUND_NEW)

    colors = cycle("bgrcmyk")

    # make graph object
    graph = nx.DiGraph()
    graph.add_nodes_from([c for c in np.arange(len(cartesian_clusters))])

    # set node label and position in graph
    labels = {}
    node_color = []
    node_border = []
    for i, cartesian_cluster_coordinates in enumerate(cartesian_clusters):
        cluster_color = next(colors)
        label = i + 1
        labels[i] = label
        node_color.append(cluster_color)
        node_border.append(BLACK)
        graph.nodes[i]["pos"] = cartesian_cluster_coordinates

    return graph, labels, node_border, node_color


def convert_geographic_to_cart(clusters: [Cluster], bound: [float]) -> [(int, int)]:
    lat_min, lat_max, lon_min, lon_max = bound
    delta_lat = lat_max - lat_min
    delta_lon = lon_max - lon_min
    zero_lat = lat_min / delta_lat
    zero_lon = lon_min / delta_lon

    output = []

    for i, cluster in enumerate(clusters):
        lat, lon = cluster.center

        y = lat / delta_lat - zero_lat
        x = lon / delta_lon - zero_lon

        output.append((x, y))

    return output
