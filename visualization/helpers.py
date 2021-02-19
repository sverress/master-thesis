from classes import Cluster
from globals import BLACK, GEOSPATIAL_BOUND_NEW
from itertools import cycle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


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


def alt_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=1.0,
    bbox=None,
    ax=None,
    rotate=True,
    **kwds,
):
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = dict(((u, v), d) for u, v, d in G.edges(data=True))
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )

        if rotate:
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360  # degrees
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0),)
        if str(label) != label:
            label = str(label)  # this will cause "1" and 1 to be labeled the same

        # set optional alignment
        horizontalalignment = kwds.get("horizontalalignment", "center")
        if horizontalalignment is list:
            horizontalalignment = horizontalalignment[
                list(labels.values()).index(label)
            ]

        verticalalignment = kwds.get("verticalalignment", "center")[
            list(labels.values()).index(label)
        ]
        if verticalalignment is list:
            verticalalignment = verticalalignment[list(labels.values()).index(label)]

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=True,
        )
        text_items[(n1, n2)] = t

    return text_items


def choose_label_alignment(start: int, end: int, pos: dict):
    start_pos = pos[start]
    end_pos = pos[end]

    if start_pos[0] <= end_pos[0]:
        return "top"
    else:
        return "bottom"


def edge_label(start: int, end: int, pos: dict, number_of_trip: int):
    start_pos = pos[start]
    end_pos = pos[end]

    if start_pos[0] <= end_pos[0]:
        return f"{start+1} --> {end+1}: {number_of_trip}"
    else:
        return f"{number_of_trip} : {end+1} <-- {start+1}"
