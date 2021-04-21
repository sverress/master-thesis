import classes
import itertools
import globals
import numpy as np
import helpers
import abc


class Decorators:
    @classmethod
    def check_setup(cls, func):
        def return_function(self, *args, **kwargs):
            if self.weights is not None:
                return func(self, *args, **kwargs)
            else:
                raise ValueError(
                    "Value function is not setup with a state. "
                    "Run value_function.setup() to initialize value function."
                )

        return return_function


class ValueFunction(abc.ABC):
    def __init__(
        self,
        weight_update_step_size=globals.WEIGHT_UPDATE_STEP_SIZE,
        weight_init_value=globals.WEIGHT_INITIALIZATION_VALUE,
        discount_factor=globals.DISCOUNT_RATE,
        vehicle_inventory_step_size=globals.VEHICLE_INVENTORY_STEP_SIZE,
    ):
        # for every location - 3 bit for each location
        # for every cluster, 1 float for deviation, 1 float for battery deficient
        # for vehicle - n bits for scooter inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # + n bits for battery inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # for every small depot - 1 float for degree of filling
        self.number_of_features_per_cluster = 3
        self.location_repetition = 3
        self.vehicle_inventory_step_size = vehicle_inventory_step_size
        self.step_size = weight_update_step_size
        self.discount_factor = discount_factor
        self.weight_init_value = weight_init_value

        self.weights = None
        self.location_indicator = None

    @abc.abstractmethod
    def setup(self, state: classes.State):
        """
        Method for setting up the value function when the state is known
        :param state: state to infer weights with
        """
        pass

    @abc.abstractmethod
    def estimate_value(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        time: int,
        state_features=None,
    ):
        pass

    @abc.abstractmethod
    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):
        pass

    @abc.abstractmethod
    def get_state_features(
        self, state: classes.State, vehicle: classes.Vehicle, time: int
    ):
        pass

    @Decorators.check_setup
    def convert_state_to_features(
        self, state: classes.State, vehicle: classes.Vehicle, time: int
    ):
        location_indicator = (
            [0] * self.location_repetition * vehicle.current_location.id
            + [1] * self.location_repetition
            + [0]
            * self.location_repetition
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
                cluster.ideal_state - len(cluster.scooters)
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
            else (
                len(vehicle.scooter_inventory)
                / (vehicle.scooter_inventory_capacity + 0.000001)
            )
        )
        battery_inventory_percent = (
            0
            if vehicle.battery_inventory_capacity == 0
            else (vehicle.battery_inventory / vehicle.battery_inventory_capacity)
        )

        scooter_inventory_indication = [
            1
            if self.vehicle_inventory_step_size * i
            < scooter_inventory_percent
            <= self.vehicle_inventory_step_size * (i + 1)
            or scooter_inventory_percent == i
            else 0
            for i in range(round(1 / self.vehicle_inventory_step_size))
        ]

        battery_inventory_indication = [
            1
            if self.vehicle_inventory_step_size * i
            < battery_inventory_percent
            <= self.vehicle_inventory_step_size * (i + 1)
            or battery_inventory_percent == i
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


class LinearValueFunction(ValueFunction):
    def setup(self, state):
        number_of_locations_indicators = len(state.locations) * self.location_repetition
        number_of_state_features = (
            (self.number_of_features_per_cluster * len(state.clusters))
            + (2 * round(1 / self.vehicle_inventory_step_size))
            + len(state.locations)
            - len(state.clusters)
            - 1
        )

        self.weights = [self.weight_init_value] * (
            number_of_locations_indicators
            + number_of_state_features
            + (number_of_locations_indicators * number_of_state_features)
        )
        self.location_indicator = [0] * number_of_locations_indicators

    @Decorators.check_setup
    def estimate_value(
        self,
        state,
        vehicle,
        time,
        state_features=None,
    ):
        if not state_features:
            state_features = self.get_state_features(state, vehicle, time)

        current_state_value = float(np.dot(self.weights, state_features))

        return current_state_value

    @Decorators.check_setup
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

    @Decorators.check_setup
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

    @Decorators.check_setup
    def get_state_features(self, state, vehicle, time):
        return self.create_location_features_combination(
            self.convert_state_to_features(state, vehicle, time)
        )
