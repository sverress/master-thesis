from classes import Action, Scooter, State, Vehicle
from visualization.helpers import *
import matplotlib.pyplot as plt
import copy
from itertools import cycle


def visualize_clustering(clusters):
    fig, ax = plt.subplots(figsize=[10, 6])

    # Add image to background
    oslo = plt.imread("images/kart_oslo.png")
    lat_min, lat_max, lon_min, lon_max = GEOSPATIAL_BOUND_NEW
    ax.imshow(
        oslo,
        zorder=0,
        extent=(lon_min, lon_max, lat_min, lat_max),
        aspect="auto",
        alpha=0.6,
    )
    colors = cycle("bgrcmyk")
    # Add clusters to figure
    for cluster in clusters:
        scooter_locations = [
            (scooter.get_lat(), scooter.get_lon()) for scooter in cluster.scooters
        ]
        cluster_color = next(colors)
        df_scatter = ax.scatter(
            [lon for lat, lon in scooter_locations],
            [lat for lat, lon in scooter_locations],
            c=cluster_color,
            alpha=0.6,
            s=3,
        )
        center_lat, center_lon = cluster.get_location()
        rs_scatter = ax.scatter(
            center_lon,
            center_lat,
            c=cluster_color,
            edgecolor="None",
            alpha=0.8,
            s=200,
        )
        ax.annotate(
            cluster.id,
            (center_lon, center_lat),
            ha="center",
            va="center",
            weight="bold",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if len(clusters) > 0:
        # Legend will use the last cluster color. Check for clusters to avoid None object
        ax.legend(
            [df_scatter, rs_scatter],
            ["Full dataset", "Cluster centers"],
            loc="upper right",
        )
    plt.show()


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


def visualize_vehicle_routes(
    state,
    current_vehicle_id=None,
    current_location_id=None,
    next_location_id=None,
    tabu_list=None,
    policy="",
):
    """
    Visualize the vehicle route in a state from a simulation
    :param policy: name of current policy
    :param tabu_list: current tabulist
    :param current_location_id: vehicles current location id
    :param state: State to display
    :param current_vehicle_id: current vehicle at a cluster
    :param next_location_id: id of next state
    :return:
    """
    fig, ax1, ax2 = create_two_subplot_fig(
        titles=[
            "Tabu list",
            f"Vehicle {current_vehicle_id} arriving at location {current_location_id} and heading to location {next_location_id}",
        ],
        fig_title=policy.__str__(),
    )

    plot_tabu_list(ax1, tabu_list)

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
    ) = setup_cluster_visualize(
        state, current_location_id, next_location_id, fig=fig, ax=ax2
    )

    if current_vehicle_id or current_vehicle_id == 0:
        route_labels, alignment = add_vehicle_routes(
            graph, node_border, state.vehicles, current_vehicle_id, next_location_id
        )

        alt_draw_networkx_edge_labels(
            graph,
            edge_labels=route_labels,
            verticalalignment=alignment,
            bbox=dict(alpha=0),
            ax=ax2,
        )

    # displays plot
    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax2)

    func = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [func("_", COLORS[i]) for i in range(len(state.vehicles))]
    legends = [f"Vehicle {vehicle.id}" for vehicle in state.vehicles]
    ax2.legend(handles, legends, framealpha=1)

    # shows the plots in IDE
    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_action(
    state_before_action: State,
    vehicle_before_action: Vehicle,
    current_state: State,
    current_vehicle: Vehicle,
    action: Action,
    scooter_label=True,
    policy="",
):
    # creating the subplots for the visualization
    fig, ax1, ax2, ax3 = create_three_subplot_fig(
        titles=["Action", "State before action", "State after action"],
        fig_title=policy.__str__(),
    )

    # plots the vehicle info and the action in the first plot
    plot_vehicle_info(vehicle_before_action, current_vehicle, ax1)
    plot_action(
        action,
        vehicle_before_action.current_location.id,
        ax1,
        offset=(
            len(vehicle_before_action.scooter_inventory)
            + len(current_vehicle.scooter_inventory)
        )
        * ACTION_OFFSET,
    )

    make_scooter_visualize(state_before_action, ax2, scooter_label=scooter_label)
    add_location_center(state_before_action.locations, ax2)

    make_scooter_visualize(current_state, ax3, scooter_label=scooter_label)
    add_location_center(state_before_action.locations, ax3)

    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_scooters_on_trip(current_state: State, trips: [(int, int, Scooter)]):
    fig, ax1, ax2 = create_two_subplot_fig(["Current trips", "State"])

    plot_trips(trips, ax1)

    make_scooter_visualize(current_state, ax2, scooter_battery=True)

    add_location_center(current_state.locations, ax2)

    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_scooter_simulation(
    current_state: State,
    trips,
):
    """
    Visualize scooter trips of one system simulation
    :param current_state: Initial state for the simulation
    :param trips: trips completed during a system simulation
    """

    # creating the subplots for the visualization
    fig, ax1, ax2, ax3 = create_three_subplot_fig(
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
    ) = make_scooter_visualize(current_state, ax2, scooter_label=True)

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


def visualize_analysis(instances, title=None):
    """
    :param instances: world instances to analyse
    :param title: plot title
    :return: plot for the analysis
    """
    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))

    # creating subplots
    spec = gridspec.GridSpec(figure=fig, ncols=2, nrows=2)

    subplots_labels = [
        ("Time", "Number of lost trips", " Lost demand"),
        ("Time", "Number of available scooters", "Total available scooters"),
        (
            "Time",
            "Avg. deficient number of scooters",
            "Negative deviation ideal state",
        ),
        ("Time", "Avg deficient battery per e-scooter (%)", "Deficient battery"),
    ]
    # figure
    subplots = []
    for i, (x_label, y_label, plot_title) in enumerate(subplots_labels):
        ax = create_plot_with_axis_labels(
            fig,
            spec[i],
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
        )
        subplots.append(ax)

    ax1, ax2, ax3, ax4 = subplots
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")

    for i, instance in enumerate(instances):
        (
            lost_demand,
            deviation_ideal_state,
            deficient_battery,
            total_available_scooters,
        ) = instance.metrics.get_all_metrics()
        x = instance.metrics.timeline

        label = (
            instance.label.split("/")[1]
            if hasattr(instance.policy, "value_function")
            else instance.label
        )

        ax1.plot(x, lost_demand, c=COLORS[i], label=label)
        ax2.plot(x, total_available_scooters, c=COLORS[i], label=label)
        ax3.plot(x, deviation_ideal_state, c=COLORS[i], label=label)
        ax4.plot(x, deficient_battery, c=COLORS[i], label=label)

    for subplot in subplots:
        subplot.legend()
        subplot.set_ylim(ymin=0)
    if title is not None:
        fig.suptitle(
            title,
            fontsize=16,
        )

    fig.tight_layout()

    plt.show()

    return fig


def visualize_td_error(td_errors_and_label: [([float], str)]):
    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))

    # creating subplots
    spec = gridspec.GridSpec(
        figure=fig, ncols=1, nrows=1, width_ratios=[1], wspace=0.2, hspace=0
    )

    ax = create_plot_with_axis_labels(
        fig,
        spec[0],
        x_label="Update",
        y_label="TD-error",
        plot_title="",
    )

    for i, (td_errors, label) in enumerate(td_errors_and_label):
        x = np.arange(len(td_errors))
        ax.plot(x, td_errors, color=COLORS[i], label=label, zorder=len(td_errors) - i)

    fig.suptitle(
        f"TD-error development",
        fontsize=16,
    )

    ax.legend()
    ax.set_xlim(xmin=0)

    plt.show()

    return fig
