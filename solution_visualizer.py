import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Model import Model

def make_graph(nodes: dict):
    graph = nx.DiGraph()
    graph.add_nodes_from([i for i in range(len(nodes.keys()))])

    labels = {}
    for i,p in enumerate(nodes.keys()):
        labels[i] = nodes[p]['label']
        graph[i]['pos'] = p

    return graph, labels

def add_vehicle_info(seed: int, number_of_vehicles, vehicles_cons, ax):
    np.random.seed(seed)
    colors = [
        "#%06X" % np.random.randint(0, 0xFFFFFF) for i in range(number_of_vehicles)
    ]

    # adding vehicle color description
    for i in range(len(colors)):
        s = "Vehicle %d" % (i + 1)
        ax.text(
            -0.25,
            1 - 0.03 * i,
            s,
            transform=ax.transAxes,
            c=colors[i],
            fontsize=10,
            weight="bold",
            verticalalignment="top",
        )

    # vehicle info box
    cons = (
            f"Vehicle constraint:\nTime = %s \n\nVehicle capacity:\nBattery = %s \nScooter = %s"
            % vehicles_cons
    )
    props = dict(boxstyle="round", facecolor="wheat", pad=0.5, alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        -0.25,
        1 - 0.03 * (len(colors) + 1),
        cons,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    return colors

# TODO change input to instace and clean up code with new
def visualize_model_solution(
    model: Model,
    node_locations: list,
    number_of_vehicles: int,
    time_cost: dict,
    reward: list,
    vehicle_cons: tuple,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9.7))
    ax1.set_title("Model solution", fontweight="bold")
    ax2.set_title("Edges not included in solution", fontweight="bold")

    colors = add_vehicle_info(10, number_of_vehicles, vehicle_cons, ax1)

    # TODO add node dicti as input - Sverre must set up model first
    graph, labels = make_graph({})

    # getting position of nodes in graph
    pos = nx.get_node_attributes(graph, 'pos')

    # adding reward to nodes
    for i in pos.keys():
        if i != 0:
            ax1.text(
                pos[i][0] + 0.12,
                pos[i][1] - 0.05,
                s="r=" + str(round(reward[i], 1)),
                weight="bold",
                horizontalalignment="left",
            )

    edge_labels = {}

    # adding edges
    # model.x returns a dictionary of all x-variables, where i_j_v is key and value is value
    for key in model.x.keys():
        if model.x[key] > 0:
        graph.add_edge(key[0], key[1], color=colors[key[2]], width=2)
        edge_labels[(key[0], key[1])] = "t = " + str(round(time_cost[(key[0], key[1])], 2))+ " | l = " + str(model.l[key])

    # displaying solution graph
    node_color = ["white" for i in range(len(node_locations))]

    # TODO Change this som it displays scooter vs delivery
    node_border = ["black" for i in range(len(node_locations))]

    edges = graph.edges()
    e_colors = [graph[u][v]["color"] for u, v in edges]
    e_weights = [graph[u][v]["width"] for u, v in edges]

    nx.draw(
        graph,
        pos,
        node_color=node_color,
        edgecolors=node_border,
        edge_color=e_colors,
        width=e_weights,
        node_size=500,
        with_labels=False,
        ax=ax1,
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax1)
    nx.draw_networkx_labels(
        graph,
        pos,
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


if __name__ == "__main__":
    m = Model()
    m.setup()
    m.optimize_model()
    visualize_model_solution(m, m.locations_coordinates, m.num_service_vehicles, m.time_cost, m.reward, (m.T_max, m.Q_b, m.Q_s))
    # m.print_solution()
