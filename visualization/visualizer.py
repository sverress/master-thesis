import matplotlib.pyplot as plt
from instance.helpers import create_sections
from visualization.helpers import *


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

    node_dict = create_node_dict(instance)
    graph, labels, node_border, node_color = make_graph(node_dict)
    pos = nx.get_node_attributes(graph, "pos")

    # adding reward and type color to nodes
    for i, p in enumerate(node_dict.keys()):  # i is number in node list
        if node_dict[p]["label"] != DEPOT:
            s = "r=" + str(round(instance.model.get_parameters().reward[i], 2))
            for k in range(instance.model_input.num_service_vehicles):
                if (i, k) in instance.model.p.keys() and instance.model.p[(i, k)].x > 0:
                    s += "\n p_%s=%s" % (k + 1, int(instance.model.p[(i, k)].x))
            x, y = pos[i]
            ax1.text(
                x + 0.035, y, s, weight="bold", horizontalalignment="left",
            )
    edge_labels = {}

    # adding edges
    for key in instance.model.x.keys():
        from_node, to_node, vehicle_id = key
        if instance.model.x[key].x > 0:
            graph.add_edge(from_node, to_node, color=colors[vehicle_id], width=2)
            edge_labels[(from_node, to_node)] = (
                "T = "
                + str(
                    round(
                        instance.model.get_parameters().time_cost[(from_node, to_node)],
                        2,
                    )
                )
                + ", L_%d = %d"
                % (vehicle_id + 1, int(instance.model.l[(from_node, vehicle_id)].x))
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
    legend_color = [BLUE, GREEN, RED]
    legend_text = ["Depot", "Scooter", "Delivery"]

    for i in range(len(legend_text)):
        ax1.scatter(1.2, 1 - 0.05 * i, s=100, c=legend_color[i], marker="o", alpha=0.7)
        ax1.annotate(legend_text[i], (1.22, 0.992 - 0.05 * i))

    # show figure
    plt.show()


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
