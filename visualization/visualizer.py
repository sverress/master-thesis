from classes import Action, State
from visualization.helpers import *
from globals import *
import matplotlib.pyplot as plt
import itertools
import copy


# TODO Fix object imports. Here Python expert Sverre can shine
def visualize_state(state, ax=None):
    """
    Visualize the clusters of a state with battery and number of scooters in the clusters
    :param state: State object to be visualized
    :param ax: Optional subplot to plot the graph on
    """
    setup_visualize(state, ax)
    # shows the plots in IDE
    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_cluster_flow(state: State, flows: [(int, int, int)], ax=None):
    """
    Visualize the flow in a state from a simulation
    :param state: State to display
    :param flows: TODO: explanation of flows format
    :param ax: Optional subplot to plot the graph on
    :return:
    """
    graph = setup_visualize(state, ax)

    # adds edges of flow between the clusters
    edge_labels, alignment = add_flow_edges(graph, flows)

    # displays edges on plot
    alt_draw_networkx_edge_labels(
        graph,
        edge_labels=edge_labels,
        verticalalignment=alignment,
        bbox=dict(alpha=0),
        ax=ax,
    )

    # shows the plots in IDE
    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_scooter_simulation(
    current_state: State, next_state: State, action: Action, trips,
):
    """
    Visualize scooter trips of one system simulation
    :param current_state: Initial state for the simulation
    :param next_state: The state after the simulation is done
    # TODO take in the state before an action is performed, so we can dispaly state before/after an action i one plot
    :param action: actions to be performed before the simulation
    :param trips: trips completed during a system simulation
    """

    node_size = 50
    font_size = 10

    # creating the subplots for the visualization
    fig, ax1, ax2, ax3 = create_system_simulation_plot()

    # plots the vehicle info and the action in the first plot
    plot_vehicle_info(current_state.vehicle, next_state.vehicle, ax1)
    plot_action(action, ax1)

    # make a list of all scooters
    all_current_scooters = list(
        itertools.chain.from_iterable(
            map(lambda cluster: cluster.scooters, current_state.clusters)
        )
    )

    # list of all scooter ids for the scooter label plot
    all_current_scooters_id = [scooter.id for scooter in all_current_scooters]
    # list of all cluster ids associated with scooters (so we can get the right color of the scooter nodes)
    all_current_cluster_ids = list(
        itertools.chain.from_iterable(
            [cluster.id] * len(cluster.scooters) for cluster in current_state.clusters
        )
    )

    # constructs the networkx graph from cluster location, second input is for color purpose
    graph, labels, node_border, node_color = make_graph(
        [scooter.get_location() for scooter in all_current_scooters],
        all_current_cluster_ids,
    )

    # add scooter id as label above each node in plot
    add_scooter_id(all_current_scooters, graph, ax2)

    # display graph
    display_graph(
        graph,
        node_color,
        node_border,
        node_size,
        labels,
        font_size,
        ax2,
        with_labels=False,
    )

    # have to copy the networkx graph since the plot isn't shown in the IDE yet
    next_graph = copy.deepcopy(graph)

    # convert location of the scooter that has moved during the simulation
    cartesian_coordinates = convert_geographic_to_cart(
        [scooter.get_location() for star, end, scooter in trips], GEOSPATIAL_BOUND_NEW
    )

    number_of_current_scooters = len(all_current_scooters)

    # adds labels to the new subplot of the scooters from the state before simulation
    add_scooter_id(all_current_scooters, next_graph, ax3)

    # loop to add nodes/scooters that have moved during a simulation
    for i, trip in enumerate(trips):
        start, end, scooter = trip
        x, y = cartesian_coordinates[i]
        previous_label = all_current_scooters_id.index(scooter.id)

        # add new node
        next_graph.add_node(number_of_current_scooters + i)
        # adds location in graph for the new node
        next_graph.nodes[number_of_current_scooters + i]["pos"] = (x, y)

        # adds label and color of new node
        labels[number_of_current_scooters + i] = previous_label
        node_color.append(COLORS[end.id])
        node_border.append(BLACK)

        # set the previous position of the scooter to a white node
        node_color[previous_label] = "white"

        # add edge from previous location of scooter to current
        next_graph.add_edge(
            previous_label, number_of_current_scooters + i, color=BLACK, width=1
        )

        # display label on subplot
        ax3.text(x, y + 0.015, f"{scooter.id}", horizontalalignment="center")

    display_graph(
        next_graph,
        node_color,
        node_border,
        node_size,
        labels,
        font_size,
        ax3,
        with_labels=False,
    )

    plt.tight_layout(pad=1.0)
    plt.show()
