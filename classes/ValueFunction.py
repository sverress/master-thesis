import classes
import itertools
from globals import *
import numpy as np


class ValueFunction:
    def __init__(
        self,
        number_of_clusters: int,
        number_of_depots=3,
        weight_update_step_size=0.1,
        discount_factor=DISCOUNT_RATE,
        vehicle_inventory_step_size=0.25,
        number_of_small_depots=2,
        number_of_features_per_cluster=2,
    ):
        # for every location - 3 bit for location
        # for every cluster 1 float for deviation, 1 float for battery deficient
        # for vehicle - n bits for scooter inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # + n bits for battery inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # for every small depot - 1 float for degree of filling
        number_of_locations_features = (number_of_clusters + number_of_depots) * 3
        number_of_state_features = (
            (number_of_features_per_cluster * number_of_clusters)
            + (2 * int(1 / vehicle_inventory_step_size))
            + number_of_small_depots
        )

        self.weights = [0] * (
            number_of_locations_features
            + number_of_state_features
            + (number_of_locations_features * number_of_state_features)
        )
        self.location_indicator = [0] * (3 * (number_of_clusters + number_of_depots))
        self.vehicle_inventory_step_size = vehicle_inventory_step_size
        self.step_size = weight_update_step_size
        self.discount_factor = discount_factor

    def estimate_value(
        self, state: classes.State, vehicle: classes.Vehicle, time: int,
    ):

        current_linear_model_features = ValueFunction.create_location_features_combination(
            self.convert_state_to_features(state, vehicle, time)
        )

        current_state_value = float(np.dot(current_linear_model_features, self.weights))

        return current_state_value

    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):
        self.weights += (
            self.step_size
            * (reward - (self.discount_factor * next_state_value) - current_state_value)
            * current_state_features
        )

    def convert_state_to_features(
        self, state: classes.State, vehicle: classes.Vehicle, time: int
    ):
        location_indicator = (
            self.location_indicator[: vehicle.current_location.id * 3]
            + [1, 1, 1]
            + self.location_indicator[(vehicle.current_location.id + 1) * 3 :]
        )

        normalized_deviation_ideal_state = ValueFunction.normalize_list(
            [
                abs(len(cluster.scooters) - cluster.ideal_state)
                for cluster in state.clusters
            ]
        )
        normalized_deficient_battery = ValueFunction.normalize_list(
            [
                len(cluster.scooters) - cluster.get_current_state()
                for cluster in state.clusters
            ]
        )

        scooter_inventory_percent = (
            len(vehicle.scooter_inventory) / vehicle.scooter_inventory_capacity
        )
        battery_inventory_percent = (
            vehicle.battery_inventory / vehicle.battery_inventory_capacity
        )

        scooter_inventory_indication = [
            1
            if int(scooter_inventory_percent / self.vehicle_inventory_step_size) == i
            else 0
            for i in range(int(1 / self.vehicle_inventory_step_size))
        ]

        battery_inventory_indication = [
            1
            if int(battery_inventory_percent / self.vehicle_inventory_step_size) == i
            else 0
            for i in range(int(1 / self.vehicle_inventory_step_size))
        ]

        small_depot_degree_of_filling = [
            depot.get_available_battery_swaps(time) / SMALL_DEPOT_CAPACITY
            for depot in state.depots[1:]
        ]

        state_features = (
            normalized_deviation_ideal_state
            + normalized_deficient_battery
            + scooter_inventory_indication
            + battery_inventory_indication
            + small_depot_degree_of_filling
        )

        return location_indicator + state_features

    def create_location_features_combination(self, state_features):
        location_indicator = state_features[: len(self.location_indicator)]
        state_features = state_features[len(self.location_indicator) :]

        locations_features_combination = list(
            itertools.chain(
                *[
                    np.multiply(indicator, state_features).tolist()
                    for indicator in location_indicator
                ]
            )
        )

        return locations_features_combination

    @staticmethod
    def normalize_list(parameter_list: [float]):
        min_value = min(parameter_list)
        max_value = max(parameter_list)

        return [
            (parameter - min_value) / (max_value - min_value)
            for parameter in parameter_list
        ]
