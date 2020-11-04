import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def make_graph(nodes: dict):
    # make graph object
    graph = nx.DiGraph()
    graph.add_nodes_from([i for i in range(len(nodes.keys()))])

    # set node label and position
    labels = {}
    node_color = []
    node_border = []
    for i, p in enumerate(nodes.keys()):
        if nodes[p]["label"] == "Depot":
            labels[i] = "D"
            node_color.append("blue")
            node_border.append("black")
        elif nodes[p]["label"] == "S":
            labels[i] = i
            node_color.append("green")
            node_border.append("black")
        elif nodes[p]["label"] == "D":
            labels[i] = i
            node_color.append("red")
            node_border.append("black")

        graph.nodes[i]["pos"] = p

    return graph, labels, node_border, node_color


def visualize_solution(instance, node_list: dict):
    # generate plot and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9.7))
    ax1.set_title("Model solution", fontweight="bold")
    ax2.set_title("Edges not included in solution", fontweight="bold")

    # add vehicle and node info to plot
    colors = add_vehicle_node_info(
        10, instance.num_service_vehicles, instance.model._.get_vehicle_cons(), ax1
    )

    graph, labels, node_border, node_color = make_graph(node_list)

    # adding reward and type color to nodes
    for i, p in enumerate(node_list.keys()):  # i is number in node list
        if node_list[p]["label"] != "Depot":
            s = "r=" + str(round(instance.model._.reward[i], 1))
            for k in range(instance.num_service_vehicles):
                if instance.model.p[(i, k)].x > 0:
                    s += "\n p_%s=%s" % (k + 1, int(instance.model.p[(i, k)].x))

            ax1.text(
                p[0] + 0.12, p[1] - 0.05, s, weight="bold", horizontalalignment="left",
            )
    edge_labels = {}

    # adding edges
    for key in instance.model.x.keys():
        if instance.model.x[key].x > 0:
            graph.add_edge(key[0], key[1], color=colors[key[2]], width=2)
            edge_labels[(key[0], key[1])] = (
                "T = "
                + str(round(instance.model._.time_cost[(key[0], key[1])], 2))
                + ", L_%d = %d"
                % (key[2] + 1, int(instance.model.l[(key[1], key[2])].x))
            )

    # set edge color for solution
    edges = graph.edges()
    e_colors = [graph[u][v]["color"] for u, v in edges]
    e_weights = [graph[u][v]["width"] for u, v in edges]

    pos = nx.get_node_attributes(graph, "pos")

    # draw solution graph
    nx.draw(
        graph,
        pos,
        node_color=node_color,
        edgecolors=node_border,
        edge_color=e_colors,
        width=e_weights,
        node_size=500,
        alpha=0.7,
        with_labels=False,
        ax=ax1,
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax1)
    nx.draw_networkx_labels(
        graph,
        pos,
        labels,
        font_size=14,
        font_color="white",
        font_weight="bold",
        ax=ax1,
    )

    # displaying graph for nodes/edges not in solution
    display_edge_plot(instance, node_list, edge_labels, ax2)

    # add description for nodes
    c = ["blue", "green", "red"]
    t = ["Depot", "Scooter", "Delivery"]

    x_max = ax1.axis()[1]
    y_max = ax1.axis()[3]

    for i in range(3):
        ax1.scatter(x_max, y_max - 0.2 * i, s=100, c=c[i], marker="o", alpha=0.7)
        ax1.annotate(t[i], (x_max + 0.08, y_max - 0.05 - 0.2 * i))

    # show figure
    plt.show()


def add_vehicle_node_info(seed: int, number_of_vehicles, vehicles_cons, ax):
    # generate random colors for vehicle routs
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
        f"Vehicle constraint:\nTime = %s \n\nCar capacity:\nBattery = %s \nScooter = %s"
        % (vehicles_cons[0], vehicles_cons[1][0], vehicles_cons[2][0])
    )

    if len(vehicles_cons[1]) > 1:
        cons += f"\n\nBike capacity:\nBattery = %s \nScooter = %s" % (
            vehicles_cons[1][1],
            vehicles_cons[2][1],
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


def display_edge_plot(instance, node_list: dict, s_edge_labels: dict, ax):
    # draw nodes
    graph, labels, node_border, node_color = make_graph(node_list)

    # draw edges and set label (time cost and inventory)
    edge_labels = {}
    for x in instance.model.x:
        if instance.model.x[x].x == 0:
            if (
                x[0] != x[1]
                and not s_edge_labels.keys().__contains__((x[0], x[1]))
                and not s_edge_labels.keys().__contains__((x[1], x[0]))
            ):
                graph.add_edge(
                    x[0], x[1], color="grey", width=1, style="dotted", alpha=0.2
                )
                edge_labels[(x[0], x[1])] = "t = " + str(
                    round(instance.model._.time_cost[(x[0], x[1])], 2)
                )

    # set node and edge color
    node_color = ["white" for i in range(len(node_list.keys()))]
    node_border = ["white" for i in range(len(node_list.keys()))]

    edges = graph.edges()
    e_colors = [graph[u][v]["color"] for u, v in edges]
    e_weights = [graph[u][v]["width"] for u, v in edges]

    pos = nx.get_node_attributes(graph, "pos")

    # draw graph
    nx.draw(
        graph,
        pos,
        node_color=node_color,
        edgecolors=node_border,
        edge_color=e_colors,
        width=e_weights,
        node_size=1,
        with_labels=False,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, labels, font_size=1, font_color="w", ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)
