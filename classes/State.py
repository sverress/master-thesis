from itertools import cycle

from classes.Cluster import Cluster
from classes.Vehicle import Vehicle
from classes.Action import Action
from math import sqrt, pi, sin, cos, atan2
import matplotlib.pyplot as plt

from globals import GEOSPATIAL_BOUND, GEOSPATIAL_BOUND_NEW


class State:
    def __init__(self, clusters: [Cluster], current: Cluster, vehicle: Vehicle):
        self.clusters = clusters
        self.current_cluster = current
        self.vehicle = vehicle
        self.distance_matrix = self.calculate_distance_matrix()

    def get_scooters(self):
        all_scooters = []
        for cluster in self.clusters:
            for scooter in cluster.scooters:
                all_scooters.append(scooter)
        return all_scooters

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
        :return: List of object Action
        """

        # Assume that no battery swap or pick-up of scooter with 100% battery and
        # that the scooters with the lowest battery are swapped and picked up
        swappable_scooters = self.current_cluster.get_swappable_scooters()

        # Initiate constraints for battery swap, pick-up and drop-off
        pick_ups = min(
            max(
                len(self.current_cluster.scooters) - self.current_cluster.ideal_state, 0
            ),
            self.vehicle.scooter_inventory_capacity,
        )
        swaps = min(len(self.current_cluster.scooters), self.vehicle.battery_inventory)
        drop_offs = max(
            min(
                self.current_cluster.ideal_state - len(self.current_cluster.scooters),
                len(self.vehicle.scooter_inventory),
            ),
            0,
        )

        combinations = []
        # Different combinations of battery swaps, pick-ups, drop-offs and clusters
        for cluster in self.clusters:
            # Next cluster cant be same as current
            if cluster == self.current_cluster:
                continue
            for pick_up in range(pick_ups + 1):
                for swap in range(swaps + 1):
                    for drop_off in range(drop_offs + 1):
                        if (pick_up + swap) <= self.vehicle.battery_inventory and (
                            pick_up + swap
                        ) <= len(self.current_cluster.scooters):
                            combinations.append([swap, pick_up, drop_off, cluster])

        actions = []
        for combination in combinations:
            actions.append(
                Action(
                    swappable_scooters[
                        combination[1] : combination[0] + combination[1]
                    ],
                    swappable_scooters[: combination[1]],
                    self.vehicle.scooter_inventory[: combination[2]],
                    combination[3],
                )
            )
        return actions

    def get_current_reward(self, action: Action):
        reward = 0
        # Reward for battery swaps
        for battery_swap_scooter in action.battery_swaps:
            reward += (100.0 - battery_swap_scooter.battery) / 100.0

        # Reward for pickups
        for pick_up_scooter in action.pick_ups:
            reward -= pick_up_scooter.battery / 100.0

        # Reward for drop-offs
        reward += 1.0 * len(action.delivery_scooters)

        return reward

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

    def visualize_clustering(self):
        fig, ax = plt.subplots(figsize=[10, 6])

        # Add image to background
        oslo = plt.imread("test_data/kart_oslo.png")
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
        for cluster in self.clusters:
            scooter_locations = [
                (scooter.lat, scooter.lon) for scooter in cluster.scooters
            ]
            cluster_color = next(colors)
            df_scatter = ax.scatter(
                [lon for lat, lon in scooter_locations],
                [lat for lat, lon in scooter_locations],
                c=cluster_color,
                alpha=0.3,
                s=3,
            )
            center_lat, center_lon = cluster.center
            rs_scatter = ax.scatter(
                center_lon,
                center_lat,
                c=cluster_color,
                edgecolor="None",
                alpha=0.5,
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

        if len(self.clusters) > 0:
            # Legend will use the last cluster color. Check for clusters to avoid None object
            ax.legend(
                [df_scatter, rs_scatter],
                ["Full dataset", "Cluster centers"],
                loc="upper right",
            )
        plt.show()
