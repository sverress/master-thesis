import numpy as np
import helpers


def filtering_neighbours(
    state,
    vehicle,
    pick_up,
    delivery,
    number_of_neighbours,
    exclude=None,
):
    has_inventory = len(vehicle.scooter_inventory) + pick_up - delivery > 0
    clusters = sorted(
        [
            cluster
            for cluster in state.clusters
            if cluster.id != vehicle.current_location.id and cluster.id not in exclude
        ],
        key=lambda cluster: len(cluster.get_available_scooters()) - cluster.ideal_state,
        reverse=not has_inventory,
    )
    has_inventory_snacks = (
        [clusters[-1]]
        if len(vehicle.scooter_inventory) + pick_up - delivery
        < vehicle.scooter_inventory_capacity
        else []
    )
    return (
        clusters[: number_of_neighbours - 1] + has_inventory_snacks
        if has_inventory
        else clusters[:number_of_neighbours]
    )


def add_depots_as_neighbours(state, time, vehicle, max_swaps):
    """
    Adds big depot and closest available (able to change all flat batteries) small depot
    """
    if vehicle.is_at_depot() or not (
        time
        and vehicle.battery_inventory - max_swaps
        < vehicle.battery_inventory_capacity * 0.2
    ):
        return []

    big_depot, *small_depots = state.depots
    # Filter out small depots that are not able to change all flat batteries for current vehicles
    available_small_depots = [
        depot
        for depot in small_depots
        if depot.get_available_battery_swaps(time) >= vehicle.flat_batteries()
    ]
    return (
        [big_depot]
        + [
            min(
                [depot for depot in available_small_depots],
                key=lambda depot: state.get_distance(
                    vehicle.current_location.id, depot.id
                ),
            )
        ]
        if available_small_depots
        else [big_depot]
    )


def get_deviation_ideal_state(state):
    # cluster score based on deviation
    return [cluster.ideal_state - len(cluster.scooters) for cluster in state.clusters]


def get_battery_deficient_in_clusters(state):
    # cluster score based on how much deficient of battery the cluster have
    return [
        len(cluster.scooters) - cluster.get_current_state()
        for cluster in state.clusters
    ]
