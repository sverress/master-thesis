import classes
from globals import SMALL_DEPOT_CAPACITY, BATTERY_LIMIT
import helpers
import abc
from scipy import stats


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
        trace_decay,
    ):
        # for every location - 3 bit for each location
        # for every cluster, 1 float for deviation, 1 float for battery deficient
        # for vehicle - n bits for scooter inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # + n bits for battery inventory in percentage ranges (e.g 0-10%, 10%-20%, etc..)
        # for every small depot - 1 float for degree of filling
        self.number_of_features_per_cluster = 2
        self.location_repetition = location_repetition
        self.vehicle_inventory_step_size = vehicle_inventory_step_size
        self.step_size = weight_update_step_size
        self.discount_factor = discount_factor
        self.weight_init_value = weight_init_value
        self.trace_decay = trace_decay

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
    def reset_eligibilities(self):
        pass

    @abc.abstractmethod
    @Decorators.check_setup
    def get_state_features(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        time: int,
        cache=None,  # current_states, available_scooters = cache
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
        cache=None,  # current_states, available_scooters = cache
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
                + len(state.locations)  # Add location features for other vehicles
            ),
        )

    def create_features(
        self,
        state,
        vehicle,
        time,
        action=None,
        cache=None,
    ):
        # Create a flag for if it is a get next state features call
        is_next_action = action is not None
        # Change location by swapping location indicator
        location_indicators = self.get_location_indicator(
            action.next_location if is_next_action else vehicle.current_location.id,
            len(state.locations),
        )
        # Fetch all normalized scooter state representations
        deviations, battery_deficiency = ValueFunction.get_normalized_lists(
            state,
            cache,
            current_location=vehicle.current_location.id if is_next_action else None,
            action=action,
        )
        # Inventory indicators adjusting for action effects
        scooter_inventory_indication = self.get_inventory_indicator(
            helpers.zero_divide(
                len(vehicle.scooter_inventory)
                + (
                    len(action.pick_ups) - len(action.delivery_scooters)
                    if is_next_action
                    else 0
                ),
                vehicle.scooter_inventory_capacity,
            )
        )

        battery_inventory_indication = self.get_inventory_indicator(
            helpers.zero_divide(
                vehicle.battery_inventory
                - (
                    len(action.battery_swaps) + len(action.pick_ups)
                    if is_next_action
                    else 0
                ),
                vehicle.battery_inventory_capacity,
            )
        )
        # Depot state
        small_depot_degree_of_filling = ValueFunction.get_small_depot_degree_of_filling(
            time, state
        )
        # Encode location of other vehicles
        other_vehicles_locations = [
            int(
                any(
                    [
                        other_vehicle.current_location.id == location.id
                        for other_vehicle in state.vehicles
                        if other_vehicle.id != vehicle.id
                    ]
                )
            )
            for location in state.locations
        ]
        return (
            location_indicators
            + deviations
            + battery_deficiency
            + scooter_inventory_indication
            + battery_inventory_indication
            + small_depot_degree_of_filling
            + other_vehicles_locations
        )

    @Decorators.check_setup
    def convert_state_to_features(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        time: int,
        cache=None,
    ):
        return self.create_features(
            state,
            vehicle,
            time,
            action=None,
            cache=cache,
        )

    def update_shifts_trained(self, shifts_trained: int):
        self.shifts_trained = shifts_trained

    @Decorators.check_setup
    def convert_next_state_features(self, state, vehicle, action, time, cache=None):
        return self.create_features(
            state,
            vehicle,
            time,
            action=action,
            cache=cache,
        )

    @staticmethod
    def get_small_depot_degree_of_filling(time, state) -> [int]:
        return [
            depot.get_available_battery_swaps(time) / SMALL_DEPOT_CAPACITY
            for depot in state.depots[1:]
        ]

    def get_location_indicator(
        self, location_id, number_of_locations, location_repetition=None
    ) -> [int]:
        location_repetition = (
            location_repetition
            if location_repetition is not None
            else self.location_repetition
        )
        return (
            [0.0] * location_repetition * location_id
            + [1.0] * location_repetition
            + [0.0] * location_repetition * (number_of_locations - 1 - location_id)
        )

    @staticmethod
    def get_normalized_lists(
        state, cache=None, current_location=None, action=None
    ) -> ([int], [int]):
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

        current_states, available_scooters = (
            cache
            if cache is not None
            else (
                [cluster.get_current_state() for cluster in state.clusters],
                [cluster.get_available_scooters() for cluster in state.clusters],
            )
        )  # Use cache if you have it

        deviation, battery_deficiency = [], []
        for i, cluster in enumerate(state.clusters):
            deviation.append(
                len(available_scooters[i])
                - cluster.ideal_state
                + (
                    scooters_added_in_current_cluster
                    if cluster.id == current_location
                    else 0
                )  # Add available scooters from action
            )
            battery_deficiency.append(
                len(cluster.scooters)
                - current_states[i]
                - (battery_percentage_added if cluster.id == current_location else 0)
            )

        return [
            stats.zscore(list_to_normalize).tolist()
            for list_to_normalize in [deviation, battery_deficiency]
        ]

    def get_inventory_indicator(self, percent) -> [int]:
        length_of_list = round(1 / self.vehicle_inventory_step_size)
        filter_list = [
            int(
                percent
                <= i * self.vehicle_inventory_step_size
                + self.vehicle_inventory_step_size
            )
            for i in range(length_of_list)
        ]
        index = filter_list.index(1)
        return [0] * index + [1] + [0] * (length_of_list - index - 1)
