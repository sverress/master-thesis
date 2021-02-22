from classes import State
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


def visualize_simulation(state: State, trips: [(int, int, int)]):
    visualize_state(state, trips)
