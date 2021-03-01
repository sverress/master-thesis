from matplotlib import gridspec
from globals import BLACK, GEOSPATIAL_BOUND_NEW, COLORS
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from classes import State


def display_graph(
    graph, node_color, node_border, node_size, labels, font_size, ax, with_labels=True
):
    """
    Displaying a networkx graph
    """
    pos = nx.get_node_attributes(graph, "pos")

    # set edge color for solution
    edges = graph.edges()
    edge_colors = [graph[u][v]["color"] for u, v in edges]
    edge_weights = [graph[u][v]["width"] for u, v in edges]

    # draw solution graph
    nx.draw(
        graph,
        pos,
        node_color=node_color,
        edgecolors=node_border,
        edge_color=edge_colors,
        width=edge_weights,
        node_size=node_size,
        alpha=0.7,
        with_labels=with_labels,
        ax=ax,
    )
    if with_labels:
        nx.draw_networkx_labels(
            graph,
            pos,
            labels,
            font_size=font_size,
            font_color="black",
            font_weight="bold",
            ax=ax,
        )


def plot_vehicle_info(current_vehicle, next_vehicle, ax):
    """
    Adds vehicle information to a subplot
    """

    # vehicle info box
    current_vehicle_info = (
        f"Current Vehicle:\n Cap - {current_vehicle.scooter_inventory_capacity}\n"
        f" Battery - {current_vehicle.battery_inventory}\n"
        f" Scooters:"
    )

    for scooter in current_vehicle.scooter_inventory:
        current_vehicle_info += f"  {scooter}\n"

    next_vehicle_info = (
        f"Next Vehicle:\n Cap - {next_vehicle.scooter_inventory_capacity}\n"
        f" Battery - {next_vehicle.battery_inventory}\n"
        f" Scooters:"
    )

    for scooter in next_vehicle.scooter_inventory:
        next_vehicle_info += f"  {scooter}\n"

    props = dict(boxstyle="round", facecolor="wheat", pad=0.5, alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        0,
        0.98,
        current_vehicle_info,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=props,
    )

    ax.text(
        0,
        0.88,
        next_vehicle_info,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=props,
    )


def plot_action(action, ax):
    """
    Adds action information to a subplot
    """
    props = dict(boxstyle="round", facecolor="wheat", pad=0.5, alpha=0.5)

    action_string = "Swaps:\n"

    for swap in action.battery_swaps:
        action_string += f"{swap}\n"

    action_string += "\nPickups:\n"

    for pick_up in action.pick_ups:
        action_string += f"{pick_up}\n"

    action_string += "\nDelivery:\n"

    for delivery in action.delivery_scooters:
        action_string += f"{delivery}\n"

    action_string += f"\nNext cluster: {action.next_cluster.id}"

    ax.text(
        0,
        0.78,
        action_string,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=props,
    )


def make_graph(coordinates: [(float, float)], cluster_ids: [int]):
    """
    Makes a networkx graph of a list of locations and uses cluster id to give the locations color
    Location/coordinates can both be clusters and scooters
    """
    cartesian_clusters = convert_geographic_to_cart(coordinates, GEOSPATIAL_BOUND_NEW)

    # make graph object
    graph = nx.DiGraph()
    graph.add_nodes_from([c for c in np.arange(len(cartesian_clusters))])

    # set node label and position in graph
    labels = {}
    node_color = []
    node_border = []
    for i, cartesian_cluster_coordinates in enumerate(cartesian_clusters):
        label = i
        labels[i] = label
        node_color.append(COLORS[cluster_ids[i]])
        node_border.append(BLACK)
        graph.nodes[i]["pos"] = cartesian_cluster_coordinates

    return graph, labels, node_border, node_color


def convert_geographic_to_cart(
    coordinates: [(float, float)], bound: (float, float, float, float)
) -> [(int, int)]:
    """
    Converts (lon,lat) -> ([0,1], [0,1]) give the bounds of lat/lon
    :param coordinates: list of coordinates to be converted
    :param bound: lat/lon bounds
    :return: list of cartesian coordinates on the interval [0,1]
    """
    lat_min, lat_max, lon_min, lon_max = bound
    delta_lat = lat_max - lat_min
    delta_lon = lon_max - lon_min
    zero_lat = lat_min / delta_lat
    zero_lon = lon_min / delta_lon

    output = []

    for lat, lon in coordinates:
        y = lat / delta_lat - zero_lat
        x = lon / delta_lon - zero_lon

        output.append((x, y))

    return output


def choose_label_alignment(start: int, end: int, pos: dict):
    """
    Helper method to set the alignment of label above or beyond an edge
    """
    start_pos = pos[start]
    end_pos = pos[end]

    if start_pos[0] <= end_pos[0]:
        return "top"
    else:
        return "bottom"


def edge_label(start: int, end: int, pos: dict, number_of_trip: int):
    """
    Construct edge label depending on the direction of the edge arrow
    """
    start_pos = pos[start]
    end_pos = pos[end]

    if start_pos[0] <= end_pos[0]:
        return f"{start} --> {end}: {number_of_trip}"
    else:
        return f"{number_of_trip} : {end} <-- {start}"


def create_standard_state_plot():
    """
    Creates standard subplot for a state with image of Oslo as background
    """
    fig = plt.figure(figsize=(20, 9.7))

    # creating subplots
    spec = gridspec.GridSpec(figure=fig, ncols=1, nrows=1)
    ax = fig.add_subplot(spec[0])
    ax.axis("off")

    oslo = plt.imread("images/kart_oslo.png")
    ax.imshow(
        oslo, zorder=0, extent=(0, 1, 0, 1), aspect="auto", alpha=0.8,
    )

    return fig, ax


def create_system_simulation_plot():
    """
    Subplot framework for the simulation visualization
    """

    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))

    oslo = plt.imread("images/kart_oslo.png")

    # creating subplots
    spec = gridspec.GridSpec(
        figure=fig, ncols=3, nrows=1, width_ratios=[1, 8, 8], wspace=0, hspace=0
    )
    ax1 = fig.add_subplot(spec[0])
    ax1.set_title(f"Action")
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.axis("off")
    ax2 = fig.add_subplot(spec[1])
    ax2.set_title(f"Current State")
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.imshow(
        oslo, zorder=0, extent=(0, 1, 0, 1), aspect="auto", alpha=0.8,
    )
    ax2.axis("off")
    ax3 = fig.add_subplot(spec[2])
    ax3.set_title(f"Next State")
    ax3.set_xlim([-0.01, 1.01])
    ax3.set_ylim([-0.01, 1.01])
    ax3.imshow(
        oslo, zorder=0, extent=(0, 1, 0, 1), aspect="auto", alpha=0.8,
    )
    ax3.axis("off")

    return fig, ax1, ax2, ax3


def add_cluster_info(state, graph, ax):
    """
    Adds cluster info to a subplot (info i set right above the node/scatter)
    """
    pos = nx.get_node_attributes(graph, "pos")
    # add number of scooters and battery label to nodes
    for i, cluster in enumerate(state.clusters):
        node_info = f"S = {cluster.number_of_scooters()} \nB = {round(cluster.get_current_state(), 1)}"
        x, y = pos[i]
        ax.annotate(
            node_info, xy=(x, y + 0.03), horizontalalignment="center", fontsize=12
        )


def add_flow_edges(graph, flows):
    """
    Adds edge flow to the networkx graph
    """
    pos = nx.get_node_attributes(graph, "pos")
    edge_labels = {}
    alignment = []
    # adding edges
    for start, end, number_of_trips in flows:
        if number_of_trips > 0:
            graph.add_edge(start, end, color=BLACK, width=2)
            alignment.append(choose_label_alignment(start, end, pos))
            edge_labels[(start, end)] = edge_label(start, end, pos, number_of_trips)

    return edge_labels, alignment


def add_scooter_id(scooters, graph, ax):
    """
    Adds scooter id above the node/scatter
    """
    pos = nx.get_node_attributes(graph, "pos")

    for i, scooter in enumerate(scooters):
        x, y = pos[i]
        ax.text(x, y + 0.015, f"{scooter.id}", horizontalalignment="center")


def alt_draw_networkx_edge_labels(
    G,
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
    """
    New method to plot edge labels with different alignment for every edge
    """
    pos = nx.get_node_attributes(G, "pos")
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


def setup_visualize(state: State, ax=None):
    node_size = 1000
    font_size = 14

    # if subplot isn't specified, construct it
    if not ax:
        fig, ax = create_standard_state_plot()

    # constructs the networkx graph from cluster location and with cluster id
    graph, labels, node_border, node_color = make_graph(
        [(cluster.get_location()) for cluster in state.clusters],
        [cluster.id for cluster in state.clusters],
    )

    # adds cluster info (#scooters and tot battery) on plot
    add_cluster_info(state, graph, ax)

    # displays plot
    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax)

    return graph
