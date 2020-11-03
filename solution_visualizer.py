import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_model_solution(
    model: gp.Model,
    node_locations: list,
    number_of_vehicles: int,
    time_cost: dict,
    reward: list,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9.7))
    ax1.set_title("Model solution", fontweight="bold")
    ax2.set_title("Edges not included in solution", fontweight="bold")

    np.random.seed(10)
    colors = [
        "#%06X" % np.random.randint(0, 0xFFFFFF) for i in range(number_of_vehicles)
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from([i for i in range(len(node_locations))])
    labels = {}

    for i in range(len(colors)):
        s = "Vehicle %d" % (i + 1)
        ax1.text(
            -0.25,
            1 - 0.03 * i,
            s,
            transform=ax1.transAxes,
            c=colors[i],
            fontsize=10,
            weight="bold",
            verticalalignment="top",
        )

    # giving nodes positions in graph and adding reward
    for i, p in enumerate(node_locations):
        if i == 0:
            labels[i] = "D"
        else:
            labels[i] = i
            ax1.text(
                p[0] + 0.12,
                p[1] - 0.05,
                s="r=" + str(round(reward[i], 1)),
                weight="bold",
                horizontalalignment="left",
            )
        graph.nodes[i]["pos"] = p

    edge_labels = {}
    # adding edges
    for var in model.getVars():
        if var.varName.startswith("x") and var.x > 0:
            t = [int(s) for s in var.varName.strip("]").split("[")[1].split(",")]
            graph.add_edge(t[0], t[1], color=colors[t[2]], width=2)
            edge_labels[(t[0], t[1])] = "t = " + str(round(time_cost[(t[0], t[1])], 2))

    # displaying solution graph
    node_color = ["white" for i in range(len(node_locations))]
    node_border = ["black" for i in range(len(node_locations))]

    edges = graph.edges()
    e_colors = [graph[u][v]["color"] for u, v in edges]
    e_weights = [graph[u][v]["width"] for u, v in edges]

    nx.draw(
        graph,
        node_locations,
        node_color=node_color,
        edgecolors=node_border,
        edge_color=e_colors,
        width=e_weights,
        node_size=500,
        with_labels=False,
        ax=ax1,
    )
    nx.draw_networkx_edge_labels(graph, node_locations, edge_labels=edge_labels, ax=ax1)
    nx.draw_networkx_labels(
        graph,
        node_locations,
        labels,
        font_size=14,
        font_color="r",
        font_weight="bold",
        ax=ax1,
    )

    # displaying graph for nodes/edges not in solution
    display_default_plot(model, node_locations, time_cost, labels, edge_labels, ax2)

    plt.show()


def display_default_plot(
    model: gp.Model,
    node_locations: list,
    time_cost: dict,
    labels: dict,
    edge_labels: dict,
    ax2,
):
    default_graph = nx.DiGraph()
    default_graph.add_nodes_from([i for i in range(len(node_locations))])

    default_edge_labels = {}
    for var in model.getVars():
        if var.varName.startswith("x") and var.x == 0:
            t = [int(s) for s in var.varName.strip("]").split("[")[1].split(",")]
            if (
                t[0] != t[1]
                and not edge_labels.keys().__contains__((t[0], t[1]))
                and not edge_labels.keys().__contains__((t[1], t[0]))
            ):
                default_graph.add_edge(
                    t[0], t[1], color="grey", width=1, style="dotted", alpha=0.2
                )
                default_edge_labels[(t[0], t[1])] = "t = " + str(
                    round(time_cost[(t[0], t[1])], 2)
                )

    node_color = ["white" for i in range(len(node_locations))]
    node_border = ["white" for i in range(len(node_locations))]

    d_edges = default_graph.edges()
    d_e_colors = [default_graph[u][v]["color"] for u, v in d_edges]
    d_e_weights = [default_graph[u][v]["width"] for u, v in d_edges]

    nx.draw(
        default_graph,
        node_locations,
        node_color=node_color,
        edgecolors=node_border,
        edge_color=d_e_colors,
        width=d_e_weights,
        node_size=1,
        with_labels=False,
        ax=ax2,
    )
    nx.draw_networkx_labels(
        default_graph, node_locations, labels, font_size=1, font_color="w", ax=ax2
    )
    nx.draw_networkx_edge_labels(
        default_graph, node_locations, edge_labels=default_edge_labels, ax=ax2
    )
