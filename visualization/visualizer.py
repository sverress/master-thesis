from classes import Action, State, Scooter
from visualization.helpers import *
from globals import *
import matplotlib.pyplot as plt
import copy


def visualize_state(state):
    """
    Visualize the clusters of a state with battery and number of scooters in the clusters
    :param state: State object to be visualized
    """
    setup_cluster_visualize(state)
    # shows the plots in IDE
    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_cluster_flow(state: State, flows: [(int, int, int)]):
    """
    Visualize the flow in a state from a simulation
    :param state: State to display
    :param flows: flow of scooter from one cluster to another
    :return:
    """
    (
        graph,
        fig,
        ax,
        graph,
        labels,
        node_border,
        node_color,
        node_size,
        font_size,
    ) = setup_cluster_visualize(state)

    if flows:
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

    # displays plot
    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax)

    # shows the plots in IDE
    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_vehicle_route(state, vehicle_route=None, next_state_id=-1):
    """
    Visualize the vehicle route in a state from a simulation
    :param state: State to display
    :param vehicle_route: passed route for the vehicle entering a
    :param next_state_id: id of next state
    :return:
    """
    (
        graph,
        fig,
        ax,
        graph,
        labels,
        node_border,
        node_color,
        node_size,
        font_size,
    ) = setup_cluster_visualize(state, next_state_id)

    if vehicle_route:
        route_labels, alignment = add_vehicle_route(graph, node_border, vehicle_route)

        alt_draw_networkx_edge_labels(
            graph,
            edge_labels=route_labels,
            verticalalignment=alignment,
            bbox=dict(alpha=0),
            ax=ax,
        )

    # displays plot
    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax)

    # shows the plots in IDE
    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_action(state_before_action: State, current_state: State, action: Action):

    # creating the subplots for the visualization
    fig, ax1, ax2, ax3 = create_system_simulation_plot(
        ["Action", "State before action", "State after action"]
    )

    # plots the vehicle info and the action in the first plot
    plot_vehicle_info(state_before_action.vehicle, current_state.vehicle, ax1)
    plot_action(
        action,
        state_before_action.current_location.id,
        ax1,
        offset=(
            len(state_before_action.vehicle.scooter_inventory)
            + len(current_state.vehicle.scooter_inventory)
        )
        * ACTION_OFFSET,
    )

    make_scooter_visualize(state_before_action, ax2, scooter_battery=True)
    add_cluster_center(state_before_action.clusters, ax2)

    make_scooter_visualize(current_state, ax3, scooter_battery=True)
    add_cluster_center(state_before_action.clusters, ax3)

    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_scooters_on_trip(current_state: State, trips: [(int, int, Scooter)]):
    fig, ax1, ax2 = create_state_trips_plot(["Current trips", "State"])

    plot_trips(trips, ax1)

    make_scooter_visualize(current_state, ax2, scooter_battery=True)

    add_cluster_center(current_state.clusters, ax2)

    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_scooter_simulation(
    current_state: State, trips,
):
    """
    Visualize scooter trips of one system simulation
    :param current_state: Initial state for the simulation
    :param trips: trips completed during a system simulation
    """

    # creating the subplots for the visualization
    fig, ax1, ax2, ax3 = create_system_simulation_plot(
        ["Trips", "Current state", "Next State"]
    )

    plot_trips(trips, ax1)

    (
        graph,
        node_color,
        node_border,
        node_size,
        labels,
        font_size,
        all_current_scooters,
        all_current_scooters_id,
    ) = make_scooter_visualize(current_state, ax2, scooter_battery=True)

    # have to copy the networkx graph since the plot isn't shown in the IDE yet
    next_graph = copy.deepcopy(graph)

    # convert location of the scooter that has moved during the simulation
    cartesian_coordinates = convert_geographic_to_cart(
        [scooter.get_location() for star, end, scooter in trips], GEOSPATIAL_BOUND_NEW
    )

    number_of_current_scooters = len(all_current_scooters)

    # adds labels to the new subplot of the scooters from the state before simulation
    add_scooter_id_and_battery(
        all_current_scooters, next_graph, ax3, scooter_battery=True
    )

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
        ax3.text(
            x, y + 0.015, f"{scooter.id}", horizontalalignment="center", fontsize=8
        )

        ax3.text(
            x,
            y - 0.02,
            f"B - {round(scooter.battery, 1)}",
            horizontalalignment="center",
            fontsize=8,
        )

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


def visualize_analysis(instances, policies, smooth_curve=True):
    """
    :param instances: world instances to analyse
    :param policies: different policies used on the world instances
    :param smooth_curve: boolean if plot of the analysis is to be smoothed out
    :return: plot for the analysis
    """
    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))

    # creating subplots
    spec = gridspec.GridSpec(
        figure=fig, ncols=3, nrows=1, width_ratios=[1] * 3, wspace=0.2, hspace=0
    )

    subplots_labels = [
        ("Time", "Number of lost trips", " Lost demand"),
        ("Time", "Avg. number of scooters - absolute value", "Deviation ideal state"),
        ("Time", "Total deficient battery in the world", "Deficient battery"),
    ]
    # figure
    subplots = []
    for i, (x_label, y_label, plot_title) in enumerate(subplots_labels):
        ax = create_plot_with_axis_labels(
            fig, spec[i], x_label=x_label, y_label=y_label, plot_title=plot_title,
        )
        ax.legend()
        ax.set_ylim(ymin=0)
        subplots.append(ax)

    ax1, ax2, ax3 = subplots
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    for i, instance in enumerate(instances):
        (
            lost_demand,
            deviation_ideal_state,
            deficient_battery,
        ) = instance.metrics.get_all_metrics()
        x = instance.metrics.get_time_array()

        ax1.plot(x, lost_demand, c=COLORS[i], label=policies[i])
        if smooth_curve:
            plot_smoothed_curve(x, deviation_ideal_state, ax2, COLORS[i], policies[i])
            plot_smoothed_curve(x, deficient_battery, ax3, COLORS[i], policies[i])
        else:
            ax2.plot(x, deviation_ideal_state, c=COLORS[i], label=policies[i])
            ax3.plot(x, deficient_battery, c=COLORS[i], label=policies[i])

    for subplot in subplots:
        subplot.legend()
        subplot.set_ylim(ymin=0)

    fig.suptitle(
        f"Sample size {SAMPLE_SIZE} - Shift duration {SHIFT_DURATION} - Number of clusters {NUMBER_OF_CLUSTERS} - "
        f"Rollouts {NUMBER_OF_ROLLOUTS} - Max number of neighbours {MAX_NUMBER_OF_NEIGHBOURS}",
        fontsize=16,
    )

    plt.show()

    return fig
