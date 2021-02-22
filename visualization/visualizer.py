from classes import State, Action, Scooter
from clustering.scripts import *
from visualization.helpers import *
import matplotlib.pyplot as plt
import itertools


def visualize_state(state: State, ax=None):
    if not ax:
        fig, ax = create_standard_state_plot()

    # constructs the networkx graph
    graph, labels, node_border, node_color = make_graph(
        [(cluster.get_location()) for cluster in state.clusters]
    )

    add_cluster_info(state, graph, ax)

    node_size = 1000
    font_size = 14

    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax)

    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_cluster_flow(state: State, trips: [(int, int, int)]):
    fig, ax = create_standard_state_plot()

    # constructs the networkx graph
    graph, labels, node_border, node_color = make_graph(
        [(cluster.get_location()) for cluster in state.clusters]
    )

    add_cluster_info(state, graph, ax)

    edge_labels, alignment = add_edges(graph, trips)

    node_size = 1000
    font_size = 14

    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax)

    alt_draw_networkx_edge_labels(
        graph,
        edge_labels=edge_labels,
        verticalalignment=alignment,
        bbox=dict(alpha=0),
        ax=ax,
    )

    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_scooter_simulation(
    current_state: State, next_state: State, action: Action, trips,
):
    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))

    oslo = plt.imread("test_data/kart_oslo.png")

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

    plot_vehicle_info(current_state.vehicle, next_state.vehicle, ax1)

    all_current_scooters = list(
        itertools.chain.from_iterable(
            map(lambda cluster: cluster.scooters, current_state.clusters)
        )
    )

    graph, labels, node_border, node_color = make_graph(
        [scooter.get_location() for scooter in all_current_scooters]
    )

    node_size = 100
    font_size = 8

    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax2)

    all_next_scooters = list(
        itertools.chain.from_iterable(
            map(lambda cluster: cluster.scooters, next_state.clusters)
        )
    )

    next_graph, next_labels, next_node_border, next_node_color = make_graph(
        [scooter.get_location() for scooter in all_next_scooters]
    )

    display_graph(
        next_graph,
        next_node_color,
        next_node_border,
        node_size,
        next_labels,
        font_size,
        ax3,
    )

    plt.tight_layout(pad=1.0)
    plt.show()
