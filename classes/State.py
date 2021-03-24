import random

from classes.Cluster import Cluster
from classes.Vehicle import Vehicle
import clustering.methods
from system_simulation.scripts import system_simulate
from visualization.visualizer import *
import decision.neighbour_filtering
import numpy as np
import math
import pickle
import os

from globals import STATE_CACHE_DIR


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
        return self.distance_matrix[start.id][end.id]

    def get_distance_id(self, start: int, end: int):
        return self.get_distance(
            self.get_cluster_by_id(start), self.get_cluster_by_id(end)
        )

    def get_distance_to_all(self, cluster_id):
        return self.distance_matrix[cluster_id]

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

    def get_max_number_of_swaps(self, cluster: Cluster):
        return min(
            min(len(cluster.scooters), self.vehicle.battery_inventory),
            cluster.ideal_state,
        )

    def get_possible_actions(
        self, number_of_neighbours=None, divide=None, random_neighbours=0
    ):
        """
        Enumerate all possible actions from the current state
        :param random_neighbours: number of random neighbours to add to the possible next location
        :param number_of_neighbours: number of neighbours to evaluate
        :param divide: number to divide by to create range increment
        :return: List of Action objects
        """

        def get_range(max_int):
            return range(
                0,
                max_int + 1,
                math.ceil((max_int / divide) if divide else 1) if max_int else 1,
            )

        # Initiate constraints for battery swap, pick-up and drop-off
        pick_ups = min(
            max(
                len(self.current_cluster.scooters) - self.current_cluster.ideal_state, 0
            ),
            self.vehicle.scooter_inventory_capacity
            - len(self.vehicle.scooter_inventory),
        )
        swaps = self.get_max_number_of_swaps(self.current_cluster)
        drop_offs = max(
            min(
                self.current_cluster.ideal_state - len(self.current_cluster.scooters),
                len(self.vehicle.scooter_inventory),
            ),
            0,
        )

        combinations = []
        # Different combinations of battery swaps, pick-ups, drop-offs and clusters
        for cluster in decision.neighbour_filtering.filtering_neighbours(
            self,
            number_of_neighbours=number_of_neighbours,
            random_neighbours=random_neighbours,
        ):
            for pick_up in get_range(pick_ups):
                for swap in get_range(swaps):
                    for drop_off in get_range(drop_offs):
                        if (pick_up + swap) <= self.vehicle.battery_inventory and (
                            pick_up + swap
                        ) <= len(self.current_cluster.scooters):
                            combinations.append([swap, pick_up, drop_off, cluster.id])

        # Assume that no battery swap or pick-up of scooters with 100% battery and
        # that the scooters with the lowest battery are prioritized
        swappable_scooters_id = [
            scooter.id for scooter in self.current_cluster.get_swappable_scooters()
        ]
        # Adding every action. Actions are the IDs of the scooters to be handled.
        return [
            Action(
                swappable_scooters_id[pick_up : battery_swap + pick_up],
                swappable_scooters_id[:pick_up],
                [scooter.id for scooter in self.vehicle.scooter_inventory][:drop_off],
                cluster_id,
            )
            for battery_swap, pick_up, drop_off, cluster_id in combinations
        ]

    def do_action(self, action: Action):
        """
        Performs an action on the state -> changing the state + calculates the reward
        :param action: Action - action to be performed on the state
        :return: float - reward for doing the action on the state
        """
        reward = 0
        # Retrieve all scooters that you can change battery on (and therefore also pick up)
        swappable_scooters = self.current_cluster.get_swappable_scooters()

        # Perform all pickups
        for pick_up_scooter_id in action.pick_ups:
            pick_up_scooter = self.current_cluster.get_scooter_from_id(
                pick_up_scooter_id
            )
            swappable_scooters.remove(pick_up_scooter)

            # Picking up scooter and adding to vehicle inventory and swapping battery
            self.vehicle.pick_up(pick_up_scooter)

            # Remove scooter from current cluster
            self.current_cluster.remove_scooter(pick_up_scooter)

        # Perform all battery swaps
        for battery_swap_scooter_id in action.battery_swaps:
            battery_swap_scooter = self.current_cluster.get_scooter_from_id(
                battery_swap_scooter_id
            )
            swappable_scooters.remove(battery_swap_scooter)

            # Calculate reward of doing the battery swap
            if reward < self.current_cluster.ideal_state:
                reward += ((100.0 - battery_swap_scooter.battery) / 100.0) * (
                    1 - self.current_cluster.prob_of_scooter_usage()
                )

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
        self.current_cluster = self.get_cluster_by_id(action.next_cluster)

        return reward

    def __repr__(self):
        return f"State: Current cluster={self.current_cluster}"

    def get_neighbours(
        self, cluster: Cluster, number_of_neighbours=None, is_sorted=True
    ):
        """
        Get sorted list of clusters closest to input cluster
        :param is_sorted: Boolean if the neighbours list should be sorted in a ascending order based on distance
        :param cluster: cluster to find neighbours for
        :param number_of_neighbours: number of neighbours to return
        :return:
        """
        neighbours = [
            state_cluster
            for state_cluster in self.clusters
            if state_cluster.id != cluster.id
        ]
        if is_sorted:
            neighbours = sorted(
                neighbours,
                key=lambda state_cluster: self.distance_matrix[cluster.id][
                    state_cluster.id
                ],
            )
        return neighbours[:number_of_neighbours] if number_of_neighbours else neighbours

    def get_cluster_by_id(self, cluster_id: int):
        matches = [cluster for cluster in self.clusters if cluster_id == cluster.id]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"There are more than one cluster ({len(matches)} clusters) matching on id {cluster_id} in this state"
            )
        else:
            raise ValueError(f"No cluster with id={cluster_id} where found")

    def system_simulate(self):
        return system_simulate(self)

    def visualize(self):
        visualize_state(self)

    def visualize_clustering(self):
        visualize_clustering(self.clusters)

    def visualize_flow(
        self, flows: [(int, int, int)],
    ):
        visualize_cluster_flow(self, flows)

    def visualize_action(self, state_after_action, action: Action):
        visualize_action(self, state_after_action, action)

    def visualize_vehicle_route(self, vehicle_trip: [int], next_state_id: int):
        visualize_vehicle_route(self, vehicle_trip, next_state_id)

    def visualize_current_trips(self, trips: [(int, int, Scooter)]):
        visualize_scooters_on_trip(self, trips)

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

    def compute_and_set_ideal_state(self, sample_scooters):
        clustering.methods.compute_and_set_ideal_state(self, sample_scooters)

    def compute_and_set_trip_intensity(self, sample_scooters):
        clustering.methods.compute_and_set_trip_intensity(self, sample_scooters)

    def sample(self, sample_size: int):
        # Filter out scooters not in sample
        sampled_scooter_ids = random.sample(
            [scooter.id for scooter in self.get_scooters()], sample_size
        )
        for cluster in self.clusters:
            cluster.scooters = [
                scooter
                for scooter in cluster.scooters
                if scooter.id in sampled_scooter_ids
            ]

    def get_random_cluster(self, exclude=None):
        return random.choice(
            [cluster for cluster in self.clusters if cluster.id != exclude.id]
            if exclude
            else self.clusters
        )
