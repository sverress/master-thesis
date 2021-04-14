import random
from classes.Location import Location
from classes.Cluster import Cluster
from classes.Depot import Depot
import clustering.methods
from system_simulation.scripts import system_simulate
from visualization.visualizer import *
import decision.neighbour_filtering
import numpy as np
import math
import pickle
import os
from globals import STATE_CACHE_DIR
import copy


class State:
    def __init__(
        self, clusters: [Cluster], depots: [Depot], vehicles=None, distance_matrix=None,
    ):
        self.clusters = clusters
        self.vehicles = vehicles
        self.depots = depots
        self.locations = self.clusters + self.depots
        if distance_matrix:
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = self.calculate_distance_matrix()
        self.simulation_scenarios = None

    def __deepcopy__(self, *args):
        new_state = State(
            copy.deepcopy(self.clusters),
            copy.deepcopy(self.depots),
            copy.deepcopy(self.vehicles),
            distance_matrix=self.distance_matrix,
        )
        new_state.simulation_scenarios = self.simulation_scenarios
        return new_state

    def get_all_locations(self):
        return self.locations

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

    def get_distance_locations(self, start: int, end: int):
        """
        Calculate distance between two clusters
        :param start: Location id
        :param end: Location id
        :return: float - distance in kilometers
        """
        return self.distance_matrix[start][end]

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

    def get_possible_actions(
        self,
        vehicle: Vehicle,
        number_of_neighbours=None,
        divide=None,
        random_neighbours=0,
        exclude=None,
        time=None,
    ):
        """
        Enumerate all possible actions from the current state
        :param time: time of the world when the actions is to be performed
        :param random_neighbours: number of random neighbours to add to the possible next location
        :param exclude: clusters to exclude from next cluster
        :param random_neighbours: number of random neighbours to add to possible next locations
        :param vehicle: vehicle to perform this action
        :param number_of_neighbours: number of neighbours to evaluate
        :param divide: number to divide by to create range increment
        :return: List of Action objects
        """
        actions = []
        if vehicle.is_at_depot():
            neighbours = decision.neighbour_filtering.filtering_neighbours(
                self,
                vehicle,
                number_of_neighbours=number_of_neighbours,
                number_of_random_neighbours=random_neighbours,
                exclude=exclude,
            )

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
                    len(vehicle.current_location.scooters)
                    - vehicle.current_location.ideal_state,
                    0,
                ),
                vehicle.scooter_inventory_capacity - len(vehicle.scooter_inventory),
            )
            swaps = vehicle.get_max_number_of_swaps()
            drop_offs = max(
                min(
                    vehicle.current_location.ideal_state
                    - len(vehicle.current_location.scooters),
                    len(vehicle.scooter_inventory),
                ),
                0,
            )

            combinations = []
            # Different combinations of battery swaps, pick-ups, drop-offs and clusters
            for cluster in decision.neighbour_filtering.filtering_neighbours(
                self,
                vehicle,
                number_of_neighbours=number_of_neighbours,
                number_of_random_neighbours=random_neighbours,
                time=time,
                exclude=exclude,
                max_swaps=max(pick_ups, swaps),
            ):
                for pick_up in get_range(pick_ups):
                    for swap in get_range(swaps):
                        for drop_off in get_range(drop_offs):
                            if (
                                (pick_up + swap) <= vehicle.battery_inventory
                                and (pick_up + swap)
                                <= len(vehicle.current_location.scooters)
                                and pick_up + swap + drop_off > 0
                            ):
                                combinations.append(
                                    [swap, pick_up, drop_off, cluster.id]
                                )

                # Assume that no battery swap or pick-up of scooters with 100% battery and
                # that the scooters with the lowest battery are prioritized
                swappable_scooters_id = [
                    scooter.id
                    for scooter in vehicle.current_location.get_swappable_scooters()
                ]
                # Adding every action. Actions are the IDs of the scooters to be handled.
                actions = [
                    Action(
                        swappable_scooters_id[pick_up : battery_swap + pick_up],
                        swappable_scooters_id[:pick_up],
                        [scooter.id for scooter in vehicle.scooter_inventory][
                            :drop_off
                        ],
                        cluster_id,
                    )
                    for battery_swap, pick_up, drop_off, cluster_id in combinations
                ]
        return actions

    def do_action(self, action: Action, vehicle: Vehicle):
        """
        Performs an action on the state -> changing the state + calculates the reward
        :param vehicle: Vehicle to perform this action
        :param action: Action - action to be performed on the state
        :return: float - reward for doing the action on the state
        """
        reward = 0
        if not vehicle.is_at_depot():
            # Perform all pickups
            for pick_up_scooter_id in action.pick_ups:
                pick_up_scooter = vehicle.current_location.get_scooter_from_id(
                    pick_up_scooter_id
                )
                # Picking up scooter and adding to vehicle inventory and swapping battery
                vehicle.pick_up(pick_up_scooter)

                # Remove scooter from current cluster
                vehicle.current_location.remove_scooter(pick_up_scooter)
            # Perform all battery swaps
            for battery_swap_scooter_id in action.battery_swaps:
                battery_swap_scooter = vehicle.current_location.get_scooter_from_id(
                    battery_swap_scooter_id
                )
                # Calculate reward of doing the battery swap
                reward += (
                    (100.0 - battery_swap_scooter.battery) / 100.0
                ) * vehicle.current_location.prob_of_scooter_usage()

                # Decreasing vehicle battery inventory
                vehicle.change_battery(battery_swap_scooter)

            # Dropping of scooters
            for delivery_scooter_id in action.delivery_scooters:
                # Rewarding 1 for delivery
                reward += 1.0

                # Removing scooter from vehicle inventory
                delivery_scooter = vehicle.drop_off(delivery_scooter_id)

                # Adding scooter to current cluster and changing coordinates of scooter
                vehicle.current_location.add_scooter(delivery_scooter)

        # Moving the state/vehicle from this to next cluster
        vehicle.set_current_location(self.get_location_by_id(action.next_location))

        return reward

    def __repr__(self):
        return (
            f"<State: {len(self.get_scooters())} scooters in {len(self.clusters)} "
            f"clusters with {len(self.vehicles)} vehicles>"
        )

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

    def visualize_action(
        self,
        vehicle_before_action: Vehicle,
        current_state: State,
        vehicle: Vehicle,
        action: Action,
        policy: str,
    ):
        visualize_action(
            self, vehicle_before_action, current_state, vehicle, action, policy
        )

    def visualize_vehicle_routes(
        self,
        current_vehicle_id: int,
        current_location_id: int,
        next_location_id: int,
        tabu_list: [int],
        policy: str,
    ):
        visualize_vehicle_routes(
            self,
            current_vehicle_id,
            current_location_id,
            next_location_id,
            tabu_list,
            policy,
        )

    def visualize_current_trips(self, trips: [(int, int, int)]):
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

    def compute_and_set_trip_intensity(
        self, sample_scooters, ideal_state_computation=False
    ):
        if ideal_state_computation:
            for cluster in self.clusters:
                cluster.trip_intensity_per_iteration = cluster.ideal_state * 0.1
        else:
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

    def get_vehicle_by_id(self, vehicle_id: int) -> Vehicle:
        """
        Returns the vehicle object in the state corresponding to the vehicle id input
        :param vehicle_id: the id of the vehicle to fetch
        :return: vehicle object
        """
        try:
            return [vehicle for vehicle in self.vehicles if vehicle_id == vehicle.id][0]
        except IndexError:
            raise ValueError(
                f"There are no vehicle in the state with an id of {vehicle_id}"
            )
