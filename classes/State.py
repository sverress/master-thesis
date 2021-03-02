from itertools import cycle
from classes.Cluster import Cluster
from classes.Vehicle import Vehicle
from classes.Action import Action
from clustering.methods import (
    compute_and_set_ideal_state,
    compute_and_set_trip_intensity,
)
from system_simulation.scripts import system_simulate
from visualization.visualizer import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from globals import GEOSPATIAL_BOUND_NEW, STATE_CACHE_DIR


class State:
    def __init__(self, clusters: [Cluster], current: Cluster, vehicle: Vehicle):
        self.clusters = clusters
        self.current_cluster = current
        self.vehicle = vehicle
        self.distance_matrix = self.calculate_distance_matrix()

    def get_cluster_by_lat_lon(self, lat: float, lon: float):
        """
        :param lat: lat location of scooter
        :param lon:
        :return:
        """
        return min(self.clusters, key=lambda cluster: cluster.distance_to(lat, lon))

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
        :return: float - distance in kilometers
        """
        if start not in self.clusters:
            raise ValueError("Start cluster not in state")
        elif end not in self.clusters:
            raise ValueError("End cluster not in state")

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
                    neighbour_distance.append(
                        cluster.distance_to(*neighbour.get_location())
                    )
            distance_matrix.append(neighbour_distance)
        return distance_matrix

    def get_possible_actions(self, number_of_neighbours=1):
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
        swappable_scooters_id = [scooter.id for scooter in swappable_scooters]

        # Initiate constraints for battery swap, pick-up and drop-off
        pick_ups = min(
            max(
                len(self.current_cluster.scooters) - self.current_cluster.ideal_state, 0
            ),
            self.vehicle.scooter_inventory_capacity
            - len(self.vehicle.scooter_inventory),
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
        for cluster in self.get_neighbours(
            self.current_cluster, number_of_neighbours=number_of_neighbours
        ):
            for pick_up in range(pick_ups + 1):
                for swap in range(swaps + 1):
                    for drop_off in range(drop_offs + 1):
                        if (pick_up + swap) <= self.vehicle.battery_inventory and (
                            pick_up + swap
                        ) <= len(self.current_cluster.scooters):
                            combinations.append([swap, pick_up, drop_off, cluster])

        actions = []
        # Only need ID of scooter to drop off.
        vehicle_inventory_id = list(
            map(lambda scooter: scooter.id, self.vehicle.scooter_inventory)
        )

        # Adding every action. Actions are the IDs of the scooters to be handled.
        for battery_swap, pick_up, drop_off, cluster in combinations:
            actions.append(
                Action(
                    swappable_scooters_id[pick_up : battery_swap + pick_up],
                    swappable_scooters_id[:pick_up],
                    vehicle_inventory_id[:drop_off],
                    cluster,
                )
            )
        return actions

    def do_action(self, action: Action):
        """
        Performs an action on the state -> changing the state + calculates the reward
        :param action: Action - action to be performed on the state
        :return: float - reward for doing the action on the state
        """
        reward = 0
        # Retrieve all scooters that you can change battery on (and therefor also pick up)
        swappable_scooters = self.current_cluster.get_swappable_scooters()

        # Perform all pickups
        for pick_up_scooter_id in action.pick_ups:
            pick_up_scooter = self.current_cluster.get_scooter_from_id(
                pick_up_scooter_id
            )
            swappable_scooters.remove(pick_up_scooter)

            reward -= pick_up_scooter.battery / 100.0

            # Picking up scooter and adding to vehicle inventory and swapping battery
            self.vehicle.pick_up(pick_up_scooter)

            # Remove scooter from current cluster
            self.current_cluster.remove_scooter(pick_up_scooter)

            # Set scooter coordinates to None
            # TODO can be moved into the pick_up function
            pick_up_scooter.set_coordinates(None, None)

        # Perform all battery swaps
        for battery_swap_scooter_id in action.battery_swaps:
            battery_swap_scooter = self.current_cluster.get_scooter_from_id(
                battery_swap_scooter_id
            )
            swappable_scooters.remove(battery_swap_scooter)

            # Calculate reward of doing the battery swap
            reward += (100.0 - battery_swap_scooter.battery) / 100.0

            # Decreasing vehicle battery inventory
            self.vehicle.change_battery(battery_swap_scooter)

        # Dropping of scooters
        for delivery_scooter_id in action.delivery_scooters:
            # Rewarding 1 for delivery
            reward += 1.0

            # Removing scooter from vehicle inventory
            delivery_scooter = self.vehicle.drop_off(delivery_scooter_id)

            # Adding scooter to current cluster and changing coordinates of scooter
            self.current_cluster.add_scooter(delivery_scooter)

        # Moving the state/vehicle from this to next cluster
        self.current_cluster = action.next_cluster

        return reward

    def __repr__(self):
        return f"State: Current cluster={self.current_cluster}"

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

        if len(self.clusters) > 0:
            # Legend will use the last cluster color. Check for clusters to avoid None object
            ax.legend(
                [df_scatter, rs_scatter],
                ["Full dataset", "Cluster centers"],
                loc="upper right",
            )
        plt.show()

    def get_neighbours(self, cluster: Cluster, number_of_neighbours=None):
        """
        Get sorted list of clusters closest to input cluster
        :param cluster: cluster to find neighbours for
        :param number_of_neighbours: number of neighbours to return
        :return:
        """
        neighbours = sorted(
            [
                state_cluster
                for state_cluster in self.clusters
                if state_cluster.id != cluster.id
            ],
            key=lambda state_cluster: self.distance_matrix[cluster.id][
                state_cluster.id
            ],
        )
        return neighbours[:number_of_neighbours] if number_of_neighbours else neighbours

    def get_cluster_by_id(self, cluster_id: int):
        clusters = [cluster for cluster in self.clusters if cluster.id == cluster_id]
        if len(clusters) > 0:
            return next(clusters)
        else:
            raise ValueError(f"State dosen't contain cluster{cluster_id}")

    def system_simulate(self):
        return system_simulate(self)

    def visualize(self):
        visualize_state(self)

    def visualize_flow(self, flows: [(int, int, int)], next_state_id: int):
        visualize_cluster_flow(self, flows, next_state_id)

    def visualize_action(self, state_after_action, action: Action):
        visualize_action(self, state_after_action, action)

    def visualize_system_simulation(self, trips):
        visualize_scooter_simulation(self, trips)

    def set_probability_matrix(self, probability_matrix: np.ndarray):
        if probability_matrix.shape != (len(self.clusters), len(self.clusters)):
            ValueError(
                f"The shape of the probability matrix does not match the number of clusters in the class:"
                f" {probability_matrix.shape} != {(len(self.clusters), len(self.clusters))}"
            )
        for cluster in self.clusters:
            cluster.set_move_probabilities(probability_matrix[cluster.id])

    def save_state(self):
        # If there is no state_cache directory, create it
        if not os.path.exists(STATE_CACHE_DIR):
            os.makedirs(STATE_CACHE_DIR)
        with open(self.get_filepath(), "wb") as file:
            pickle.dump(self, file)

    def get_filepath(self):
        return (
            f"{STATE_CACHE_DIR}/c{len(self.clusters)}s{len(self.get_scooters())}.pickle"
        )

    @classmethod
    def load_state(cls, filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def compute_and_set_ideal_state(self, sample_size=None):
        compute_and_set_ideal_state(self, sample_size=sample_size)

    def compute_and_set_trip_intensity(self, sample_size=None):
        compute_and_set_trip_intensity(self, sample_size=sample_size)
