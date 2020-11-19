import errno
import os
import time
from matplotlib import gridspec
import matplotlib.pyplot as plt
from instance.helpers import create_sections
from visualization.helpers import *


def visualize_solution(
    instance, save, edge_plot=False, time_stamp=time.strftime("%d-%m %H.%M")
):
    """
    Visualize a solution from the model. The visualization is divided into two frames.
    Frame one: All nodes (with corresponding reward and p-value, if its picked up),
    directed edges (with corresponding time to travel that edges as well as inventory for vehicle i on that edge)
    info about vehicles
    Frame two: Edges that are not included in solution and corresponding time to travel that edge
    :param save: bool - if model should be displayed or saved
    :param instance: Instance object for a given solution
    :param edge_plot: bool - true if edge plot should be displayed
    :param time_stamp: str - time stamp when instance manager was created (to get the right folder to save figure in)
    """

    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))
    fig.tight_layout(pad=1.0)

    # removed second plot, but stored if we want to use it for later.
    if edge_plot:
        spec = gridspec.GridSpec(
            ncols=2, nrows=1, width_ratios=[1, 8, 8], wspace=0, hspace=0
        )
        ax1 = fig.add_subplot(spec[0])
        ax1.axis("off")
        ax2 = fig.add_subplot(spec[1])
        ax3 = fig.add_subplot(spec[2])
        ax2.set_title("Model solution", fontweight="bold")
        ax3.set_title("Edges not included in solution", fontweight="bold")
    else:
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 16])
        ax1 = fig.add_subplot(spec[0])
        ax1.axis("off")
        ax2 = fig.add_subplot(spec[1])
        ax2.set_title("Model solution", fontweight="bold")

    # add vehicle and node info to plot
    colors = add_vehicle_node_info(instance, ax1)

    node_dict = create_node_dict(instance)
    graph, labels, node_border, node_color = make_graph(node_dict, instance.bound)
    pos = nx.get_node_attributes(graph, "pos")

    # check to handle infeasible models
    if instance.is_feasible():

        # adding reward and type color to nodes
        for i, p in enumerate(node_dict.keys()):  # i is number in node list
            if node_dict[p]["label"] != DEPOT:
                s = ""
                pad = 0
                if node_dict[p]["label"] == SUPPLY:
                    s += "B=" + str(int(instance.model_input.B[i] * 100)) + "\n"
                for k in range(instance.model_input.num_service_vehicles):
                    if (
                        (i, k) in instance.model.p.keys()
                        and instance.model.p[(i, k)].x > 0
                        and instance.model.y[(i, k)].x > 0
                    ):
                        s += "P_%s=%s" % (k + 1, int(instance.model.p[(i, k)].x))
                        pad = 0.012
                x, y = pos[i]
                ax2.text(
                    x - 0.01,
                    y + pad + 0.01,
                    s,
                    fontsize=8,
                    weight="bold",
                    horizontalalignment="left",
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
                            instance.model.get_parameters().time_cost[
                                (from_node, to_node)
                            ],
                            2,
                        )
                    )
                    + ", "
                    + "L_%d = %d"
                    % (vehicle_id + 1, int(instance.model.l[(to_node, vehicle_id)].x))
                )

        # second plot for nodes/edges not in solution
        if edge_plot:
            display_edge_plot(instance, ax3, edge_labels)

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
            node_size=300,
            alpha=0.7,
            with_labels=False,
            ax=ax2,
        )

        nx.draw_networkx_edge_labels(
            graph, pos, edge_labels=edge_labels, font_size=8, ax=ax2
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels,
            font_size=10,
            font_color="white",
            font_weight="bold",
            ax=ax2,
        )

    else:
        ax2.set_title("Model is Infeasible", fontweight="bold")

    # add description for nodes
    legend_color = [BLUE, GREEN, RED]
    legend_text = ["Depot", "Scooter", "Delivery"]

    for i in range(len(legend_text)):
        ax2.scatter(1.01, 1 - 0.05 * i, s=100, c=legend_color[i], marker="o", alpha=0.7)
        ax2.annotate(legend_text[i], (1.017, 0.993 - 0.05 * i))

    # adding zones to plot
    add_zones(instance.number_of_sections, ax2)

    # show or save figure
    if save:
        try:
            os.makedirs(f"saved_models_fig/{time_stamp}")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        plt.tight_layout()
        plt.savefig(f"saved_models_fig/{time_stamp}/{instance.get_model_name()}.png")
    else:
        plt.tight_layout()
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

    ax.scatter(scooters["lon"], scooters["lat"], zorder=1, alpha=0.8, s=10)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    sections, cords = create_sections(num_of_sections, bound)
    ax.set_xticks(sections["lon"])
    ax.set_yticks(sections["lat"])
    ax.grid()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    ax.scatter(
        delivery_nodes["lon"], delivery_nodes["lat"], zorder=1, alpha=0.8, s=10, c="r",
    )

    oslo = plt.imread("test_data/oslo.png")
    ax.imshow(
        oslo,
        zorder=0,
        extent=(lon_min, lon_max, lat_min, lat_max),
        aspect="equal",
        alpha=0.6,
    )

    plt.show()
