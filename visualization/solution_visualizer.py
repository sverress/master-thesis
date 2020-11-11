import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from instance.helpers import create_sections


# Global variables
depot, supply, delivery = "Depot", "S", "D"
blue, green, red, black = "blue", "green", "red", "black"


def make_graph(nodes: dict):
    """
    Creates a networkx graph of the input nodes. Adds label to the nodes
    :param nodes: dictionary of nodes [lat, lon]: "label"
    :return: networkx graph, list of node labels, list of nodes border color, list of nodes color
    """
    # Converts geographical coordinates to cartesian with lim [0,1] for visualization reasons
    nodes = convert_geographic_to_cart(nodes)

    # make graph object
    graph = nx.DiGraph()
    graph.add_nodes_from([i for i in range(len(nodes.keys()))])

    # set node label and position in graph
    labels = {}
    node_color = []
    node_border = []
    for i, p in enumerate(nodes.keys()):
        if nodes[p]["label"] == depot:
            labels[i] = delivery  # most likely, you want like it Sverre
            node_color.append(blue)
            node_border.append(black)
        elif nodes[p]["label"] == supply:
            labels[i] = i
            node_color.append(green)
            node_border.append(black)
        elif nodes[p]["label"] == delivery:
            labels[i] = i
            node_color.append(red)
            node_border.append(black)

        graph.nodes[i]["pos"] = p

    return graph, labels, node_border, node_color


def visualize_solution(instance):
    """
    Visualize a solution from the model. The visualization is divided into two frames.
    Frame one: All nodes (with corresponding reward and p-value, if its picked up),
    directed edges (with corresponding time to travel that edges as well as inventory for vehicle i on that edge)
    info about vehicles
    Frame two: Edges that are not included in solution and corresponding time to travel that edge
    :param instance: Instance object for a given solution
    """

    # generate plot and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9.7))
    ax1.set_title("Model solution", fontweight="bold")
    ax2.set_title("Edges not included in solution", fontweight="bold")

    # add vehicle and node info to plot
    colors = add_vehicle_node_info(instance, ax1)

    node_dict = instance.create_node_dict()
    graph, labels, node_border, node_color = make_graph(node_dict)
    pos = nx.get_node_attributes(graph, "pos")

    # adding reward and type color to nodes
    for i, p in enumerate(node_dict.keys()):  # i is number in node list
        if node_dict[p]["label"] != depot:
            s = "r=" + str(round(instance.model.get_parameters().reward[i], 2))
            for k in range(instance.num_service_vehicles):
                if instance.model.p[(i, k)].x > 0:
                    s += "\n p_%s=%s" % (k + 1, int(instance.model.p[(i, k)].x))
            ax1.text(
                pos[i][0] + 0.035,
                pos[i][1],
                s,
                weight="bold",
                horizontalalignment="left",
            )
    edge_labels = {}

    # adding edges
    for key in instance.model.x.keys():
        if instance.model.x[key].x > 0:
            graph.add_edge(key[0], key[1], color=colors[key[2]], width=2)
            edge_labels[(key[0], key[1])] = (
                "T = "
                + str(
                    round(
                        instance.model.get_parameters().time_cost[(key[0], key[1])], 2
                    )
                )
                + ", L_%d = %d"
                % (key[2] + 1, int(instance.model.l[(key[1], key[2])].x))
            )

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
        node_size=400,
        alpha=0.7,
        with_labels=False,
        ax=ax1,
    )
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=8, ax=ax1
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels,
        font_size=14,
        font_color="white",
        font_weight="bold",
        ax=ax1,
    )

    # second plot for nodes/edges not in solution
    display_edge_plot(instance, edge_labels, ax2)

    # add description for nodes
    legend_color = ["blue", "green", "red"]
    legend_text = ["Depot", "Scooter", "Delivery"]

    for i in range(3):
        ax1.scatter(1.2, 1 - 0.05 * i, s=100, c=legend_color[i], marker="o", alpha=0.7)
        ax1.annotate(legend_text[i], (1.22, 0.992 - 0.05 * i))

    # show figure
    plt.show()


def add_vehicle_node_info(instance, ax):
    """
    Function to add information about vehicles for the first plot
    :param instance: Instance object for a given solution
    :param ax: Subplot to plot the information
    :return: Colors corresponding to vehicles used to color edges
    """
    # generate random colors for vehicle routs
    np.random.seed(10)
    colors = [
        "#%06X" % np.random.randint(0, 0xFFFFFF)
        for i in range(instance.model_input.num_service_vehicles)
    ]

    # adding vehicle color description
    for i in range(len(colors)):
        if i < instance.service_vehicles["car"][0]:
            s = "Vehicle %d (%s)" % (i + 1, "Car")
        else:
            s = "Vehicle %d (%s)" % (i + 1, "Bike")
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
        f"Vehicle constraint:\nTime = %d h %d m \n\nCar capacity:\nBattery = %d \nScooters = %d"
        % (
            int(instance.model.get_parameters().T_max / 60),
            instance.model.get_parameters().T_max % 60,
            instance.service_vehicles["car"][2],
            instance.service_vehicles["car"][1],
        )
    )

    if instance.service_vehicles["bike"][0] > 0:
        cons += f"\n\nBike capacity:\nBattery = %d \nScooters = %d" % (
            instance.service_vehicles["bike"][2],
            instance.service_vehicles["bike"][1],
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


def display_edge_plot(instance, s_edge_labels: dict, ax):
    """
    Function to display second plot of edges not included in solution
    :param instance: Instance object for a given solution
    :param s_edge_labels: Dictionary of edges used in solution
    :param ax: Subplot
    """

    ax.axis("off")
    # draw nodes
    node_dict = instance.create_node_dict()
    graph, labels, node_border, node_color = make_graph(node_dict)

    # draw edges and set label (time cost and inventory)
    edge_labels = {}
    for x in instance.model.x:
        if instance.model.x[x].x == 0:
            if (
                x[0] != x[1]
                and not s_edge_labels.keys().__contains__((x[0], x[1]))
                and not s_edge_labels.keys().__contains__((x[1], x[0]))
            ):
                graph.add_edge(x[0], x[1], color="grey", width=1, alpha=0.2)
                edge_labels[(x[0], x[1])] = "t = " + str(
                    round(instance.model.get_parameters().time_cost[(x[0], x[1])], 2)
                )

    # set node and edge color
    node_color = ["white" for i in range(len(node_dict.keys()))]
    node_border = ["white" for i in range(len(node_dict.keys()))]

    edges = graph.edges()
    e_colors = [graph[u][v]["color"] for u, v in edges]
    e_weights = [graph[u][v]["width"] for u, v in edges]

    pos = nx.get_node_attributes(graph, "pos")

    # draw graph
    edges = nx.draw_networkx_edges(
        graph, pos, edge_color=e_colors, width=e_weights, node_size=1, ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, labels, font_size=1, font_color="w", ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)

    for e in edges:
        e.set_linestyle("dashed")


def convert_geographic_to_cart(nodes):
    """
    Function to convert geographical coordinates to cartesian
    :param nodes: Dictionary of nodes [lat,lon]: type
    :return: Dictionary of nodes [cart_x, cart_y]: type
    """
    lat = [x[0] for x in nodes.keys()]
    lon = [x[1] for x in nodes.keys()]
    delta_lat = max(lat) - min(lat)
    delta_lon = max(lon) - min(lon)
    zero_lat = min(lat) / delta_lat
    zero_lon = min(lon) / delta_lon
    output = {}

    for i, j in nodes.keys():
        key = ((i / delta_lat - zero_lat), (j / delta_lon - zero_lon))
        output[key] = nodes[(i, j)]

    return output


def visualize_test_instance(scooters, delivery_nodes, bound, num_of_sections):
    """
    Generates a visual representation of the scooters and delivery nodes with a map in the background
    :param scooters: dataframe for location of scooters
    :param delivery_nodes: dataframe for location of delivery nodes
    :param bound: tuple of lat_min, lat_max, lon_min, lon_max defining the the rectangle to look at
    :param num_of_sections: number of section in each dimension
    """
    lat_min, lat_max, lon_min, lon_max = bound
    fig, ax = plt.subplots()

    ax.scatter(scooters["lon"], scooters["lat"], zorder=1, alpha=0.4, s=10)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    sections, cords = create_sections(num_of_sections, bound)
    ax.set_xticks(sections["lon"])
    ax.set_yticks(sections["lat"])
    ax.grid()

    ax.scatter(
        delivery_nodes["lon"], delivery_nodes["lat"], zorder=1, alpha=0.4, s=10, c="r",
    )

    oslo = plt.imread("test_data/oslo.png")
    ax.imshow(
        oslo,
        zorder=0,
        extent=(lon_min, lon_max, lat_min, lat_max),
        aspect="equal",
        alpha=0.4,
    )

    plt.show()
