import classes
import itertools
import globals
import numpy as np
import helpers


class GradientDescent:
    def __init__(
        self,
        number_of_locations: int,
        number_of_clusters: int,
        weight_update_step_size=globals.WEIGHT_UPDATE_STEP_SIZE,
        discount_factor=globals.DISCOUNT_RATE,
        vehicle_inventory_step_size=globals.VEHICLE_INVENTORY_STEP_SIZE,
    ):
        # for every location - 3 bit for location
        # for every cluster 1 float for deviation, 1 float for battery deficient
        # for vehicle - n bits for scooter inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # + n bits for battery inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # for every small depot - 1 float for degree of filling
        self.number_of_features_per_cluster = 3

        number_of_locations_indicators = (
            number_of_locations * self.number_of_features_per_cluster
        )
        number_of_state_features = (
            (self.number_of_features_per_cluster * number_of_clusters)
            + (2 * round(1 / vehicle_inventory_step_size))
            + number_of_locations
            - number_of_clusters
            - 1
        )

        self.weights = [0.1] * (
            number_of_locations_indicators
            + number_of_state_features
            + (number_of_locations_indicators * number_of_state_features)
        )
        self.location_indicator = [0] * number_of_locations_indicators
        self.vehicle_inventory_step_size = vehicle_inventory_step_size
        self.step_size = weight_update_step_size
        self.discount_factor = discount_factor

    def estimate_value(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        time: int,
        state_features=None,
    ):
        if not state_features:
            state_features = self.get_state_features(state, vehicle, time)

        current_state_value = float(np.dot(state_features, self.weights))

        return current_state_value

    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):
        self.weights += np.multiply(
            self.step_size
            * (
                reward + (self.discount_factor * next_state_value) - current_state_value
            ),
            current_state_features,
        )

    def convert_state_to_features(
        self, state: classes.State, vehicle: classes.Vehicle, time: int
    ):
        location_indicator = (
            [0] * self.number_of_features_per_cluster * vehicle.current_location.id
            + [1] * self.number_of_features_per_cluster
            + [0]
            * self.number_of_features_per_cluster
            * (len(state.locations) - 1 - vehicle.current_location.id)
        )

        normalized_deviation_ideal_state_positive = helpers.normalize_list(
            [
                len(cluster.scooters) - cluster.ideal_state
                if len(cluster.scooters) - cluster.ideal_state > 0
                else 0
                for cluster in state.clusters
            ]
        )

        normalized_deviation_ideal_state_negative = helpers.normalize_list(
            [
                len(cluster.scooters) - cluster.ideal_state
                if len(cluster.scooters) - cluster.ideal_state < 0
                else 0
                for cluster in state.clusters
            ]
        )

        normalized_deficient_battery = helpers.normalize_list(
            [
                len(cluster.scooters) - cluster.get_current_state()
                for cluster in state.clusters
            ]
        )

        scooter_inventory_percent = (
            0
            if vehicle.scooter_inventory_capacity == 0
            else (len(vehicle.scooter_inventory) / vehicle.scooter_inventory_capacity)
            + 0.000001
        )
        battery_inventory_percent = (
            0
            if vehicle.battery_inventory_capacity == 0
            else (vehicle.battery_inventory / vehicle.battery_inventory_capacity)
            + 0.000001
        )

        scooter_inventory_indication = [
            1
            if round(scooter_inventory_percent / self.vehicle_inventory_step_size) == i
            else 0
            for i in range(round(1 / self.vehicle_inventory_step_size))
        ]

        battery_inventory_indication = [
            1
            if round(battery_inventory_percent / self.vehicle_inventory_step_size) == i
            else 0
            for i in range(round(1 / self.vehicle_inventory_step_size))
        ]

        small_depot_degree_of_filling = [
            depot.get_available_battery_swaps(time) / globals.SMALL_DEPOT_CAPACITY
            for depot in state.depots[1:]
        ]

        state_features = (
            normalized_deviation_ideal_state_positive
            + normalized_deviation_ideal_state_negative
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

        return location_indicator + state_features + locations_features_combination

    def get_state_features(self, state, vehicle, time):
        return self.create_location_features_combination(
            self.convert_state_to_features(state, vehicle, time)
        )
