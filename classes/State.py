from classes.Cluster import Cluster
from classes.Vehicle import Vehicle
from classes.Action import Action
from math import sqrt, pi, sin, cos, atan2


class State:
    def __init__(self, clusters: [Cluster], current: Cluster, vehicle: Vehicle):
        self.clusters = clusters
        self.current_cluster = current
        self.vehicle = vehicle
        self.distance_matrix = self.calculate_distance_matrix()

    def get_distance(self, start: Cluster, end: Cluster):
        """
        Calculate distance between two clusters
        :param start: Cluster object
        :param end: Cluster object
        :return: int - distance in kilometers
        """
        start_index = self.clusters.index(start)
        end_index = self.clusters.index(end)

        return self.distance_matrix[start_index][end_index]

    def calculate_distance_matrix(self):
        """
        Computes distance matrix for all clusters
        :return: Distance matrix
        """
        distance_matrix = []
        for cluster in self.clusters:
            neighbour_distance = []
            for neighbour in self.clusters:
                if cluster == neighbour:
                    neighbour_distance.append(0.0)
                else:
                    cluster_center_lat, cluster_center_lon = cluster.center
                    neighbour_center_lat, neighbour_center_lon = neighbour.center
                    neighbour_distance.append(
                        State.haversine(
                            cluster_center_lat,
                            cluster_center_lon,
                            neighbour_center_lat,
                            neighbour_center_lon,
                        )
                    )
            distance_matrix.append(neighbour_distance)
        return distance_matrix

    def get_possible_actions(self):
        """
        Need to figure out what actions we want to look at. The combination of pick-ups, battery swaps,
        drop-offs and next cluster is too large to try them all.
        Improvements: - Travel only to neighbouring clusters.
        - Only perform some battery swaps, eg. even numbers
        :param state: current state, State
        :return: List of object Action
        """
        actions = []

        current_cluster = self.current_cluster
        # Assume that no battery swap or pick-up of scooter with 100% battery and
        # that the scooters with the lowest battery are swapped and picked up
        swappable_scooters = current_cluster.get_swappable_scooters()

        # Different combinations of battery swaps, pick-ups, drop-offs and clusters
        for cluster in self.clusters:
            # Next cluster cant be same as current
            if cluster == current_cluster:
                continue
            # Edge case: Add action with no swap, pick-up or drop-off
            actions.append(Action([], [], [], cluster))
            if current_cluster.number_of_possible_pickups() == 0:
                # Battery swap and drop-off
                battery_counter = 0
                while (
                    battery_counter < self.vehicle.battery_inventory
                    and battery_counter < len(swappable_scooters)
                ):
                    # Edge case: No drop-offs, but all combinations of battery swaps and clusters.
                    actions.append(
                        Action(
                            [swappable_scooters[i] for i in range(battery_counter + 1)],
                            [],
                            [],
                            cluster,
                        )
                    )
                    battery_counter += 1
                delivery_counter = 0
                while (
                    delivery_counter < len(self.vehicle.scooter_inventory)
                    and delivery_counter + current_cluster.number_of_scooters()
                    < current_cluster.ideal_state
                ):
                    # Edge case: No pick-ups and battery swaps, but all combinations of delivery scooters and next cluster
                    actions.append(
                        Action(
                            [],
                            [],
                            [
                                self.vehicle.scooter_inventory[i]
                                for i in range(delivery_counter + 1)
                            ],
                            cluster,
                        )
                    )
                    # All possible battery swap combinations, combined with drop-off and next cluster combinations
                    battery_counter = 0
                    while (
                        battery_counter < self.vehicle.battery_inventory
                        and battery_counter < len(swappable_scooters)
                    ):
                        actions.append(
                            Action(
                                [
                                    swappable_scooters[i]
                                    for i in range(battery_counter + 1)
                                ],
                                [],
                                [
                                    self.vehicle.scooter_inventory[i]
                                    for i in range(delivery_counter + 1)
                                ],
                                cluster,
                            )
                        )
                        battery_counter += 1

                    delivery_counter += 1

            # Battery swap and pick-up
            else:

                battery_counter = 0
                while (
                    battery_counter < self.vehicle.battery_inventory
                    and battery_counter < len(swappable_scooters)
                ):
                    # Edge case: No pick-ups, but all combinations of battery swaps and clusters.
                    actions.append(
                        Action(
                            [swappable_scooters[i] for i in range(battery_counter + 1)],
                            [],
                            [],
                            cluster,
                        )
                    )
                    battery_counter += 1

                pick_up_counter = 0
                while (
                    pick_up_counter < self.vehicle.battery_inventory
                    and pick_up_counter < len(swappable_scooters)
                    and pick_up_counter
                    < (len(current_cluster.scooters) - current_cluster.ideal_state)
                ):
                    # Edge case: No battery swaps, but all combinations of pick-ups and clusters.
                    actions.append(
                        Action(
                            [],
                            [swappable_scooters[i] for i in range(pick_up_counter + 1)],
                            [],
                            cluster,
                        )
                    )
                    # Combinations of battery swaps, pick-ups and clusters
                    # Pick up the scooters with lowest battery, swap the next lowest.
                    battery_counter = pick_up_counter + 1
                    while (
                        battery_counter < self.vehicle.battery_inventory
                        and battery_counter < len(swappable_scooters)
                    ):
                        actions.append(
                            Action(
                                [
                                    swappable_scooters[i]
                                    for i in range(
                                        pick_up_counter + 1, battery_counter + 1
                                    )
                                ],
                                [
                                    swappable_scooters[i]
                                    for i in range(pick_up_counter + 1)
                                ],
                                [],
                                cluster,
                            )
                        )
                        battery_counter += 1
                    pick_up_counter += 1

        return actions

    def get_current_reward(self, action: Action):
        return 1

    def __str__(self):
        return f"State: Current cluster={self.current_cluster}"

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """
        Compute the distance between two points in meters
        :param lat1: Coordinate 1 lat
        :param lon1: Coordinate 1 lon
        :param lat2: Coordinate 2 lat
        :param lon2: Coordinate 2 lon
        :return: Kilometers between coordinates
        """
        radius = 6378.137
        d_lat = lat2 * pi / 180 - lat1 * pi / 180
        d_lon = lon2 * pi / 180 - lon1 * pi / 180
        a = sin(d_lat / 2) * sin(d_lat / 2) + cos(lat1 * pi / 180) * cos(
            lat2 * pi / 180
        ) * sin(d_lon / 2) * sin(d_lon / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = radius * c
        return distance
