import classes
from globals import SMALL_DEPOT_CAPACITY, BATTERY_LIMIT
import helpers
import abc


class Decorators:
    @classmethod
    def check_setup(cls, func):
        def return_function(self, *args, **kwargs):
            if self.setup_complete:
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
        weight_update_step_size,
        weight_init_value,
        discount_factor,
        vehicle_inventory_step_size,
        location_repetition,
    ):
        # for every location - 3 bit for each location
        # for every cluster, 1 float for deviation, 1 float for battery deficient
        # for vehicle - n bits for scooter inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # + n bits for battery inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # for every small depot - 1 float for degree of filling
        self.number_of_features_per_cluster = 3
        self.location_repetition = location_repetition
        self.vehicle_inventory_step_size = vehicle_inventory_step_size
        self.step_size = weight_update_step_size
        self.discount_factor = discount_factor
        self.weight_init_value = weight_init_value

        self.setup_complete = False
        self.location_indicator = None
        self.shifts_trained = 0
        self.td_errors = []

    def compute_and_record_td_error(
        self,
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):
        td_error = (
            reward + (self.discount_factor * next_state_value) - current_state_value
        )
        self.td_errors.insert(len(self.td_errors), td_error)
        return td_error

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def setup(self, state: classes.State):
        """
        Method for setting up the value function when the state is known
        :param state: state to infer weights with
        """
        self.setup_complete = True

    @abc.abstractmethod
    @Decorators.check_setup
    def estimate_value(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        time: int,
    ):
        pass

    @abc.abstractmethod
    @Decorators.check_setup
    def estimate_value_from_state_features(self, state_features: [float]):
        pass

    @abc.abstractmethod
    @Decorators.check_setup
    def batch_update_weights(
        self,
        state_features,
        batch: [(float, float, float)],
    ):
        pass

    @abc.abstractmethod
    @Decorators.check_setup
    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):
        pass

    @abc.abstractmethod
    @Decorators.check_setup
    def get_state_features(
        self, state: classes.State, vehicle: classes.Vehicle, time: int
    ):
        pass

    @abc.abstractmethod
    @Decorators.check_setup
    def get_next_state_features(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        action: classes.Action,
        time: int,
    ):
        pass

    def get_number_of_location_indicators_and_state_features(
        self, state: classes.State
    ):
        return (
            len(state.locations) * self.location_repetition,
            (
                (self.number_of_features_per_cluster * len(state.clusters))
                + (2 * round(1 / self.vehicle_inventory_step_size))
                + len(state.locations)
                - len(state.clusters)
                - 1
            ),
        )

    @Decorators.check_setup
    def convert_state_to_features(
        self, state: classes.State, vehicle: classes.Vehicle, time: int
    ):
        location_indicator = self.get_location_indicator(
            vehicle.current_location.id, len(state.locations)
        )

        (
            normalized_deviation_ideal_state_positive,
            normalized_deviation_ideal_state_negative,
            normalized_deficient_battery,
        ) = ValueFunction.get_normalized_lists(state)

        scooter_inventory_indication = self.get_inventory_indicator(
            helpers.zero_divide(
                len(vehicle.scooter_inventory), vehicle.scooter_inventory_capacity
            )
        )

        battery_inventory_indication = self.get_inventory_indicator(
            helpers.zero_divide(
                vehicle.battery_inventory, vehicle.battery_inventory_capacity
            )
        )

        small_depot_degree_of_filling = ValueFunction.get_small_depot_degree_of_filling(
            time, state
        )

        return (
            location_indicator
            + normalized_deviation_ideal_state_positive
            + normalized_deviation_ideal_state_negative
            + normalized_deficient_battery
            + scooter_inventory_indication
            + battery_inventory_indication
            + small_depot_degree_of_filling
        )

    def update_shifts_trained(self, shifts_trained: int):
        self.shifts_trained = shifts_trained

    @Decorators.check_setup
    def convert_next_state_features(self, state, vehicle, action, time):
        # Change location by swapping location indicator
        location_indicators = self.get_location_indicator(
            action.next_location, len(state.locations)
        )
        # Fetch all normalized scooter state representations
        (
            normalized_deviation_ideal_state_positive,
            normalized_deviation_ideal_state_negative,
            normalized_deficient_battery,
        ) = ValueFunction.get_normalized_lists(
            state, current_location=vehicle.current_location.id, action=action
        )
        # Inventory indicators adjusting for action effects
        scooter_inventory_indication = self.get_inventory_indicator(
            helpers.zero_divide(
                len(vehicle.scooter_inventory)
                + len(action.pick_ups)
                - len(action.delivery_scooters),
                vehicle.scooter_inventory_capacity,
            )
        )

        battery_inventory_indication = self.get_inventory_indicator(
            helpers.zero_divide(
                vehicle.battery_inventory
                - len(action.battery_swaps)
                - len(action.pick_ups),
                vehicle.battery_inventory_capacity,
            )
        )
        # Depot state
        small_depot_degree_of_filling = ValueFunction.get_small_depot_degree_of_filling(
            time, state
        )
        return (
            location_indicators
            + normalized_deviation_ideal_state_positive
            + normalized_deviation_ideal_state_negative
            + normalized_deficient_battery
            + scooter_inventory_indication
            + battery_inventory_indication
            + small_depot_degree_of_filling
        )

    @staticmethod
    def get_small_depot_degree_of_filling(time, state):
        return [
            depot.get_available_battery_swaps(time) / SMALL_DEPOT_CAPACITY
            for depot in state.depots[1:]
        ]

    def get_location_indicator(self, location_id, number_of_locations):
        return (
            [0] * self.location_repetition * location_id
            + [1] * self.location_repetition
            + [0] * self.location_repetition * (number_of_locations - 1 - location_id)
        )

    @staticmethod
    def get_normalized_lists(state, current_location=None, action=None):
        if current_location is not None and action is not None:

            def filter_scooter_ids(ids, isAvailable=True):
                if isAvailable:
                    available_filter = (
                        lambda scooter_id: state.clusters[current_location]
                        .get_scooter_from_id(scooter_id)
                        .battery
                        > BATTERY_LIMIT
                    )
                else:
                    available_filter = (
                        lambda scooter_id: state.clusters[current_location]
                        .get_scooter_from_id(scooter_id)
                        .battery
                        < BATTERY_LIMIT
                    )
                return [
                    scooter_id for scooter_id in ids if available_filter(scooter_id)
                ]

            scooters_added_in_current_cluster = (
                len(
                    filter_scooter_ids(action.battery_swaps, isAvailable=False)
                )  # Add swapped scooters that where unavailable
                + len(action.delivery_scooters)  # Add delivered scooters
                - len(
                    filter_scooter_ids(action.pick_ups, isAvailable=True)
                )  # subtract removed available scooters
            )
            battery_percentage_added = sum(
                [
                    (
                        100
                        - state.clusters[current_location]
                        .get_scooter_from_id(scooter_id)
                        .battery
                    )
                    / 100
                    for scooter_id in action.battery_swaps
                ]
            ) + sum(
                [
                    (
                        100
                        - state.clusters[current_location]
                        .get_scooter_from_id(scooter_id)
                        .battery
                    )
                    / 100
                    for scooter_id in action.pick_ups
                ]
            )
        else:
            scooters_added_in_current_cluster = 0
            battery_percentage_added = 0

        def ideal_state_deviation(is_positive):
            deviations = [
                len(cluster.get_available_scooters())
                - cluster.ideal_state
                + (
                    scooters_added_in_current_cluster
                    if cluster.id == current_location
                    else 0
                )  # Add available scooters from action
                for cluster in state.clusters
            ]
            if is_positive:
                return [max(deviation, 0) for deviation in deviations]
            else:
                return [abs(min(deviation, 0)) for deviation in deviations]

        return (
            helpers.normalize_list(ideal_state_deviation(is_positive=True)),
            helpers.normalize_list(ideal_state_deviation(is_positive=False)),
            helpers.normalize_list(
                [
                    len(cluster.scooters)
                    - cluster.get_current_state()
                    - (
                        battery_percentage_added
                        if cluster.id == current_location
                        else 0
                    )
                    for cluster in state.clusters
                ]
            ),
        )

    def get_inventory_indicator(self, percent):
        return [
            1
            if self.vehicle_inventory_step_size * i
            < percent
            <= self.vehicle_inventory_step_size * (i + 1)
            or percent == i
            else 0
            for i in range(round(1 / self.vehicle_inventory_step_size))
        ]
