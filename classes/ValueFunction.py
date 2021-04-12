import classes
from globals import *
import numpy as np


class ValueFunction:
    def __init__(
        self,
        number_of_clusters: int,
        number_of_depots=3,
        weight_update_step_size=0.1,
        discount_factor=DISCOUNT_RATE,
        vehicle_inventory_step_size=0.1,
        number_of_small_depots=2,
        number_of_features_per_cluster=2,
    ):
        # for every cluster - 3 bit for location
        # all features below are multiplied by 2 since we want a feature combining location and all other features
        # for every cluster 1 float for deviation, 1 float for battery deficient
        # for vehicle - n bits for scooter inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # + n bits for battery inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # for every small depot - 1 float for degree of filling
        self.weights = [0] * (
            number_of_clusters
            * 3
            * 2
            * (
                number_of_features_per_cluster
                + (2 * int(1 / vehicle_inventory_step_size))
                + number_of_small_depots
            )
        )
        self.location_indicator = [0] * (number_of_clusters + number_of_depots)
        self.vehicle_inventory_step_size = vehicle_inventory_step_size
        self.step_size = weight_update_step_size
        self.discount_factor = discount_factor

    def estimate_value(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        action: classes.Action,
        time: int,
    ):
        current_state_features = self.convert_state_to_features(state, vehicle, time)

        current_state_value = float(np.dot(current_state_features, self.weights))

        action_distance = state.get_distance_id(
            vehicle.current_location.id, action.next_location
        )

        action_time = action.get_action_time(action_distance)

        reward = state.do_action(action, vehicle)

        next_state_features = self.convert_state_to_features(
            state, vehicle, action_time
        )

        next_state_value = float(np.dot(next_state_features, self.weights))

        self.update_weights(
            current_state_features, current_state_value, next_state_value, reward
        )

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

        locations_features_combination = np.multiply(location_indicator, state_features)

        return location_indicator + state_features + locations_features_combination

    @staticmethod
    def normalize_list(parameter_list: [float]):
        min_value = min(parameter_list)
        max_value = max(parameter_list)

        return [
            (parameter - min_value) / (max_value - min_value)
            for parameter in parameter_list
        ]
