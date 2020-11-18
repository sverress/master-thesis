import numpy as np
import networkx as nx
from itertools import product


# Global variables
DEPOT, SUPPLY, DELIVERY = "Depot", "S", "D"
BLUE, GREEN, RED, BLACK = "blue", "green", "red", "black"


def get_label(instance, i: int):
    """
    Returns the label of a specified index of a model
    :param instance: instance object
    :param i:
    :return:
    """
    if i == 0:
        return DEPOT
    if 0 < i <= instance.model_input.num_scooters:
        return SUPPLY
    else:
        return DELIVERY


def create_node_dict(instance):
    # TODO: should be moved to solution visualization script. this should return a list. Needs documentation
    output = {}
    locations = (
        [instance.depot]
        + list(zip(instance.scooters["lat"], instance.scooters["lon"]))
        + list(zip(instance.delivery_nodes["lat"], instance.delivery_nodes["lon"]))
    )
    for i, index in enumerate(locations):
        output[index] = {"label": get_label(instance, i)}
    return output


def make_graph(nodes: dict, bound):
    """
    Creates a networkx graph of the input nodes. Adds label to the nodes
    :param nodes: dictionary of nodes [lat, lon]: "label"
    :return: networkx graph, list of node labels, list of nodes border color, list of nodes color
    """
    # Converts geographical coordinates to cartesian with lim [0,1] for visualization reasons
    nodes = convert_geographic_to_cart(nodes, bound)

    # make graph object
    graph = nx.DiGraph()
    graph.add_nodes_from([i for i in range(len(nodes.keys()))])

    # set node label and position in graph
    labels = {}
    node_color = []
    node_border = []
    for i, p in enumerate(nodes.keys()):
        label = nodes[p]["label"]
        if label == DEPOT:
            labels[i] = DELIVERY
            node_color.append(BLUE)
            node_border.append(BLACK)
        elif label == SUPPLY:
            labels[i] = i
            node_color.append(GREEN)
            node_border.append(BLACK)
        elif label == DELIVERY:
            labels[i] = i
            node_color.append(RED)
            node_border.append(BLACK)

        graph.nodes[i]["pos"] = p

    return graph, labels, node_border, node_color


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

    num_of_cars, car_scooter_cap, car_battery_cap = instance.service_vehicles["car"]
    num_of_bikes, bike_scooter_cap, bike_battery_cap = instance.service_vehicles["bike"]

    # adding vehicle color description
    for i in range(len(colors)):
        if i < num_of_cars:
            s = "Vehicle %d (%s)" % (i + 1, "Car")
        else:
            s = "Vehicle %d (%s)" % (i + 1, "Bike")
        ax.text(
            -0.05,
            1 - 0.03 * i,
            s,
            transform=ax.transAxes,
            c=colors[i],
            fontsize=10,
            weight="bold",
            horizontalalignment="left",
            verticalalignment="top",
        )

    # vehicle info box
    cons = (
        f"Vehicle constraint:\nTime = %d h %d m \n\nCar capacity:\nBattery = %d \nScooters = %d"
        % (
            int(instance.model.get_parameters().T_max / 60),
            instance.model.get_parameters().T_max % 60,
            car_battery_cap,
            car_scooter_cap,
        )
    )

    if num_of_bikes > 0:
        cons += f"\n\nBike capacity:\nBattery = %d \nScooters = %d" % (
            bike_battery_cap,
            bike_scooter_cap,
        )

    props = dict(boxstyle="round", facecolor="wheat", pad=0.5, alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        -0.05,
        1 - 0.03 * (len(colors) + 1),
        cons,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=props,
    )

    return colors


def display_edge_plot(instance, ax, s_edge_labels={}):
    """
    Function to display second plot of edges not included in solution
    :param instance: Instance object for a given solution
    :param s_edge_labels: Dictionary of edges used in solution, default empty for infeasible solutions
    :param ax: Subplot
    """

    ax.axis("off")
    # draw nodes
    node_dict = create_node_dict(instance)
    graph, labels, node_border, node_color = make_graph(node_dict, instance.bound)

    edge_labels = {}

    # check to handle infeasible models
    if instance.is_feasible():
        # draw edges and set label (time cost and inventory)
        for x in instance.model.x:
            from_node, to_node, vehicle_id = x
            if instance.model.x[x].x == 0:
                if (
                    from_node != to_node
                    and not s_edge_labels.keys().__contains__((from_node, to_node))
                    and not s_edge_labels.keys().__contains__((to_node, from_node))
                ):
                    graph.add_edge(from_node, to_node, color="grey", width=1, alpha=0.2)
                    edge_labels[(from_node, to_node)] = "t = " + str(
                        round(
                            instance.model.get_parameters().time_cost[
                                (from_node, to_node)
                            ],
                            2,
                        )
                    )
    else:
        for x in instance.model.x:
            from_node, to_node, vehicle_id = x
            if (
                vehicle_id == 0
                and instance.model.get_parameters().time_cost[(from_node, to_node)] > 0
            ):
                graph.add_edge(from_node, to_node, color="grey", width=1, alpha=0.2)
                edge_labels[(from_node, to_node)] = "t = " + str(
                    round(
                        instance.model.get_parameters().time_cost[(from_node, to_node)],
                        2,
                    )
                )

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


def convert_geographic_to_cart(nodes, bound):
    """
    Function to convert geographical coordinates to cartesian
    :param nodes: Dictionary of nodes [lat,lon]: type
    :return: Dictionary of nodes [cart_x, cart_y]: type
    """
    lat_min, lat_max, lon_min, lon_max = bound
    delta_lat = lat_max - lat_min
    delta_lon = lon_max - lon_min
    zero_lat = lat_min / delta_lat
    zero_lon = lon_min / delta_lon

    output = {}

    for i, j in nodes.keys():
        key = ((j / delta_lon - zero_lon), (i / delta_lat - zero_lat))
        output[key] = nodes[(i, j)]

    return output


def add_zones(number_of_zones, ax):
    """
    Function to add zones to solution plot
    :param number_of_zones: int - number of per axis
    :param ax: subplot
    """
    axis_interval = float(1 / number_of_zones)
    xy = list(
        product(
            np.arange(axis_interval, 1, axis_interval),
            np.arange(axis_interval, 1, axis_interval),
        )
    )
    for x, y in xy:
        ax.axhline(x, xmax=0.93, color="black")
        ax.axvline(y, ymax=0.95, color="black")
