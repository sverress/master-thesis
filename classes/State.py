import random
from typing import Union
from classes.Location import Location
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
from globals import GEOSPATIAL_BOUND_NEW, STATE_CACHE_DIR


class State:
    def __init__(
        self,
        clusters: [Cluster],
        depots: [Depot],
        current_location: Union[Cluster, Depot],
        vehicle: Vehicle,
    ):
        self.clusters = clusters
        self.depots = depots
        self.locations = self.clusters + self.depots
        self.current_location = current_location
        self.vehicle = vehicle
        self.distance_matrix = self.calculate_distance_matrix()

    def get_all_locations(self):
        return self.locations

    def is_at_depot(self):
        return isinstance(self.current_location, Depot)

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

    def get_distance_locations(self, start: Location, end: Location):
        """
        Calculate distance between two clusters
        :param start: Cluster object
        :param end: Cluster object
        :return: float - distance in kilometers
        """
        if start not in self.locations:
            raise ValueError("Start cluster not in state")
        elif end not in self.locations:
            raise ValueError("End cluster not in state")
        return self.distance_matrix[start.id][end.id]

    def get_distance_id(self, start: int, end: int):
        return self.get_distance_locations(
            self.get_location_by_id(start), self.get_location_by_id(end)
        )

    def get_distance_to_all(self, location_id):
        return self.distance_matrix[location_id]

    def get_distance_to_all_clusters(self, location_id):
        return self.distance_matrix[location_id][: len(self.clusters)]

    def calculate_distance_matrix(self):
        """
        Computes distance matrix for all clusters
        :return: Distance matrix
        """
        distance_matrix = []
        for location in self.locations:
            neighbour_distance = []
            for neighbour in self.locations:
                if location == neighbour:
                    neighbour_distance.append(0.0)
                else:
                    neighbour_distance.append(
                        location.distance_to(*neighbour.get_location())
                    )
            distance_matrix.append(neighbour_distance)
        return distance_matrix

    def get_max_number_of_swaps(self, cluster: Cluster):
        return min(
            min(len(cluster.scooters), self.vehicle.battery_inventory),
            len(self.current_location.get_swappable_scooters()),
        )

    def get_possible_actions(
        self, number_of_neighbours=None, divide=None, random_neighbours=0, time=None
    ):
        """
        Enumerate all possible actions from the current state
        :param time: time of the world when the actions is to be performed
        :param random_neighbours: number of random neighbours to add to the possible next location
        :param number_of_neighbours: number of neighbours to evaluate
        :param divide: number to divide by to create range increment
        :return: List of Action objects
        """
        if self.is_at_depot():
            neighbours = decision.neighbour_filtering.filtering_neighbours(
                self,
                number_of_neighbours=number_of_neighbours,
                number_of_random_neighbours=random_neighbours,
            )
            actions = []

            for neighbour in neighbours:
                actions.append(Action([], [], [], neighbour.id))

        else:

            def get_range(max_int):
                return range(
                    0,
                    max_int + 1,
                    math.ceil((max_int / divide) if divide else 1) if max_int else 1,
                )

            # Initiate constraints for battery swap, pick-up and drop-off
            pick_ups = min(
                max(
                    len(self.current_location.scooters)
                    - self.current_location.ideal_state,
                    0,
                ),
                self.vehicle.scooter_inventory_capacity
                - len(self.vehicle.scooter_inventory),
            )
            swaps = self.get_max_number_of_swaps(self.current_location)
            drop_offs = max(
                min(
                    self.current_location.ideal_state
                    - len(self.current_location.scooters),
                    len(self.vehicle.scooter_inventory),
                ),
                0,
            )

            combinations = []
            # Different combinations of battery swaps, pick-ups, drop-offs and clusters
            for cluster in decision.neighbour_filtering.filtering_neighbours(
                self,
                number_of_neighbours=number_of_neighbours,
                number_of_random_neighbours=random_neighbours,
                time=time,
            ):
                for pick_up in get_range(pick_ups):
                    for swap in get_range(swaps):
                        for drop_off in get_range(drop_offs):
                            if (pick_up + swap) <= self.vehicle.battery_inventory and (
                                pick_up + swap
                            ) <= len(self.current_location.scooters):
                                combinations.append(
                                    [swap, pick_up, drop_off, cluster.id]
                                )

            # Assume that no battery swap or pick-up of scooters with 100% battery and
            # that the scooters with the lowest battery are prioritized
            swappable_scooters_id = [
                scooter.id for scooter in self.current_location.get_swappable_scooters()
            ]
            # Adding every action. Actions are the IDs of the scooters to be handled.
            actions = [
                Action(
                    swappable_scooters_id[pick_up : battery_swap + pick_up],
                    swappable_scooters_id[:pick_up],
                    [scooter.id for scooter in self.vehicle.scooter_inventory][
                        :drop_off
                    ],
                    cluster_id,
                )
                for battery_swap, pick_up, drop_off, cluster_id in combinations
            ]
        return actions

    def do_action(self, action: Action):
        """
        Performs an action on the state -> changing the state + calculates the reward
        :param action: Action - action to be performed on the state
        :return: float - reward for doing the action on the state
        """
        reward = 0
        if not self.is_at_depot():
            # Retrieve all scooters that you can change battery on (and therefore also pick up)
            swappable_scooters = self.current_location.get_swappable_scooters()

            # Perform all pickups
            for pick_up_scooter_id in action.pick_ups:
                pick_up_scooter = self.current_location.get_scooter_from_id(
                    pick_up_scooter_id
                )
                swappable_scooters.remove(pick_up_scooter)

                # Picking up scooter and adding to vehicle inventory and swapping battery
                self.vehicle.pick_up(pick_up_scooter)

                # Remove scooter from current cluster
                self.current_location.remove_scooter(pick_up_scooter)

            # Perform all battery swaps
            for battery_swap_scooter_id in action.battery_swaps:
                battery_swap_scooter = self.current_location.get_scooter_from_id(
                    battery_swap_scooter_id
                )
                swappable_scooters.remove(battery_swap_scooter)

                # Calculate reward of doing the battery swap
                if reward < self.current_location.ideal_state:
                    reward += (
                        (100.0 - battery_swap_scooter.battery) / 100.0
                    ) * self.current_location.prob_of_scooter_usage()

                    # Decreasing vehicle battery inventory
                    self.vehicle.change_battery(battery_swap_scooter)

            # Dropping of scooters
            for delivery_scooter_id in action.delivery_scooters:
                # Rewarding 1 for delivery
                reward += 1.0

                # Removing scooter from vehicle inventory
                delivery_scooter = self.vehicle.drop_off(delivery_scooter_id)

                # Adding scooter to current cluster and changing coordinates of scooter
                self.current_location.add_scooter(delivery_scooter)

            # Moving the state/vehicle from this to next cluster
        self.current_location = self.get_location_by_id(action.next_location)

        return reward

    def __repr__(self):
        return f"State: Current location={self.current_location}"

    def get_neighbours(
        self, location: Location, number_of_neighbours=None, is_sorted=True
    ):
        """
        Get sorted list of clusters closest to input cluster
        :param is_sorted: Boolean if the neighbours list should be sorted in a ascending order based on distance
        :param location: location to find neighbours for
        :param number_of_neighbours: number of neighbours to return
        :return:
        """
        neighbours = [
            state_location
            for state_location in self.locations
            if state_location.id != location.id
        ]
        if is_sorted:
            neighbours = sorted(
                [
                    state_location
                    for state_location in self.locations
                    if state_location.id != location.id
                ],
                key=lambda state_location: self.distance_matrix[location.id][
                    state_location.id
                ],
            )
        return neighbours[:number_of_neighbours] if number_of_neighbours else neighbours

    def get_location_by_id(self, location_id: int):
        matches = [
            location for location in self.locations if location_id == location.id
        ]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"There are more than one location ({len(matches)} locations) matching on id {location_id} in this state"
            )
        else:
            raise ValueError(f"No locations with id={location_id} where found")

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

    def visualize_vehicle_route(self, vehicle_trip: [int], next_location_id: int):
        visualize_vehicle_route(self, vehicle_trip, next_location_id)

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
