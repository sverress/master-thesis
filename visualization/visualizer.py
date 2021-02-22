from classes import State, Action, Scooter
from clustering.scripts import *
from matplotlib import gridspec
from visualization.helpers import *
import matplotlib.pyplot as plt
import networkx as nx


def visualize_state(state: State, trips=None):
    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))
    fig.tight_layout(pad=1.0)

    # add subplot to figure
    spec = gridspec.GridSpec(ncols=1, nrows=1)
    ax1 = fig.add_subplot(spec[0])
    ax1.axis("off")

    oslo = plt.imread("test_data/kart_oslo.png")
    ax1.imshow(
        oslo, zorder=0, extent=(0, 1, 0, 1), aspect="auto", alpha=0.8,
    )

    # constructs the networkx graph
    graph, labels, node_border, node_color = make_graph(state.clusters)
    pos = nx.get_node_attributes(graph, "pos")

    # add number of scooters and battery label to nodes
    for i, cluster in enumerate(state.clusters):
        node_info = f"S = {cluster.number_of_scooters()} \nB = {round(cluster.get_current_state(), 1)}"
        x, y = pos[i]
        ax1.annotate(
            node_info, xy=(x, y + 0.03), horizontalalignment="center", fontsize=12
        )

    edge_labels = {}
    alignment = []
    if trips:
        # adding edges
        for start, end, number_of_trips in trips:
            if number_of_trips > 0:
                graph.add_edge(start, end, color=BLACK, width=2)
                alignment.append(choose_label_alignment(start, end, pos))
                edge_labels[(start, end)] = edge_label(start, end, pos, number_of_trips)

    # set edge color for solution
    edges = graph.edges()
    e_colors = [graph[u][v]["color"] for u, v in edges]
    e_weights = [graph[u][v]["width"] for u, v in edges]

    # draw solution graph
    nx.draw(
        graph,
        pos,
        node_color=node_color,
        edgecolors=node_border,
        edge_color=e_colors,
        width=e_weights,
        node_size=1000,
        alpha=0.7,
        with_labels=False,
        ax=ax1,
    )

    if trips:
        alt_draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=10,
            verticalalignment=alignment,
            bbox=dict(alpha=0),
            ax=ax1,
        )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels,
        font_size=16,
        font_color="white",
        font_weight="bold",
        ax=ax1,
    )

    plt.tight_layout()
    plt.show()


def visualize_cluster_flow(state: State, trips: [(int, int, int)]):
    visualize_state(state, trips)


def visualize_simulation(current_state: State, next_state: State, action: Action):
    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))
    fig.tight_layout(pad=1.0)

    # creating subplots
    spec = gridspec.GridSpec(
        ncols=3, nrows=1, width_ratios=[1, 8, 8], wspace=0, hspace=0
    )
    ax1 = fig.add_subplot(spec[0])
    ax1.set_title(f"Action")
    ax1.axis("off")
    ax2 = fig.add_subplot(spec[1])
    ax2.set_title(f"Current State")
    ax2.axis("off")
    ax3 = fig.add_subplot(spec[2])
    ax3.set_title(f"Next State")
    ax3.axis("off")

    plot_vehicle_info(current_state.vehicle, next_state.vehicle, ax1)

    plt.show()


def plot_vehicle_info(current_vehicle, next_vehicle, ax):
    # vehicle info box
    current_vehicle_info = f"Current Vehicle:\n B_I:{current_vehicle.battery_inventory}"

    props = dict(boxstyle="round", facecolor="wheat", pad=0.5, alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        0,
        0.9,
        current_vehicle_info,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=props,
    )
