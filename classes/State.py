from classes.Action import Action
from classes.Cluster import Cluster
from classes.Vehicle import Vehicle
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
                        self.haversin(
                            cluster_center_lat,
                            cluster_center_lon,
                            neighbour_center_lat,
                            neighbour_center_lon,
                        )
                    )
            distance_matrix.append(neighbour_distance)
        return distance_matrix

    def get_possible_actions(self):
        return Action([], [], [], Cluster([]))

    def do_action(self, action: Action):
        """
        Performs an action on the state -> changing the state + calculates the reward
        :param action: Action - action to be performed on the state
        :return: int - reward for doing the action on the state
        """
        reward = 0
        # Retrieve all scooters that you can change battery on (and therefor also pick up)
        swappable_scooters = self.current_cluster.get_swappable_scooters()

        # Perform all pickups
        for pick_up_scooter in action.pick_ups:
            swappable_scooters.remove(pick_up_scooter)

            # Adding scooter to vehicle inventory
            capacity_check = self.vehicle.pick_up(pick_up_scooter)
            if not capacity_check:
                raise ValueError("Can't pick up an scooter when the vehicle is full")

            # Swap battery on scooter that is picked up
            reward -= pick_up_scooter.battery / 100.0
            pick_up_scooter.swap_battery()

            # Remove scooter from current cluster
            scooter_in_cluster = self.current_cluster.remove_scooter(pick_up_scooter)
            if not scooter_in_cluster:
                raise ValueError(
                    "Can't remove a scooter from a cluster its not current in"
                )

        # Perform all battery swaps
        for battery_swap_scooter in action.battery_swaps:
            swappable_scooters.remove(battery_swap_scooter)

            # Calculate reward of doing the battery swap
            reward += (100.0 - battery_swap_scooter.battery) / 100.0

            # Performing the battery swap
            battery_swap_scooter.swap_battery()

        for delivery_scooter in action.delivery_scooters:
            # Rewarding 1 for delivery
            reward += 1.0

            # Removing scooter from vehicle inventory
            scooter_in_vehicle = self.vehicle.deliver_scooter(delivery_scooter)

            # Error if scooter not i vehicle inventory
            if not scooter_in_vehicle:
                raise ValueError("Can't deliver a scooter that ")

            # Adding scooter to current cluster
            self.current_cluster.add_scooter(delivery_scooter)

        # Moving the state/vehicle from this to next cluster
        self.current_cluster = action.next_cluster

        return reward

    def __str__(self):
        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i+1}:")
            print(cluster.__str__() + "\n")

    @staticmethod
    def haversin(lat1, lon1, lat2, lon2):
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
