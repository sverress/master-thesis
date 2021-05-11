from .abstract import *
import numpy as np
from decision.value_functions.ANN import ANN


class ANNValueFunction(ValueFunction):
    def __init__(
        self,
        weight_update_step_size,
        weight_init_value,
        discount_factor,
        vehicle_inventory_step_size,
        location_repetition,
        trace_decay,
        network_structure: [int],
    ):
        super().__init__(
            weight_update_step_size,
            weight_init_value,
            discount_factor,
            vehicle_inventory_step_size,
            location_repetition,
            trace_decay,
        )
        self.network_structure = network_structure
        self.model = None

    def setup(self, state: classes.State):
        if self.setup_complete:
            return
        (
            number_of_locations_indicators,
            number_of_state_features,
        ) = self.get_number_of_location_indicators_and_state_features(state)
        self.model = ANN(
            self.network_structure,
            number_of_locations_indicators + number_of_state_features,
            self.trace_decay,
            self.discount_factor,
        )
        super(ANNValueFunction, self).setup(state)

    def estimate_value(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        time: int,
    ):

        return self.estimate_value_from_state_features(
            self.get_state_features(state, vehicle, time)
        )

    def estimate_value_from_state_features(self, state_features: [float]):
        return self.model.predict(state_features)

    def batch_update_weights(self):
        td_errors = []
        targets = []
        state_features = []

        for (
            current_state_value,
            next_state_value,
            reward,
            state_feature,
        ) in self.training_case_base:
            td_error = self.compute_and_record_td_error(
                current_state_value, next_state_value, reward
            )
            td_errors.append(td_error)
            targets.append(td_error + current_state_value)
            state_features.append(state_feature)

        self.model.fit(np.array(state_features), np.array(targets), td_errors)

    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):
        td_error = self.compute_and_record_td_error(
            current_state_value, next_state_value, reward
        )
        self.model.fit(
            np.array([current_state_features]),
            np.array([td_error + current_state_value]),
            td_error,
            epochs=10,
        )

    def reset_eligibilities(self):
        self.model.reset_eligibilities()

    def get_next_state_features(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        action: classes.Action,
        time: int,
        cache=None,  # current_states, available_scooters = cache
    ):
        return self.convert_next_state_features(state, vehicle, action, time, cache)

    def get_state_features(
        self, state: classes.State, vehicle: classes.Vehicle, time: int, cache=None
    ):
        return self.convert_state_to_features(state, vehicle, time, cache=cache)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = self.model.convert_model_to_string()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = ANN.load_model_from_string(state["model"])

    def __str__(self):
        return f"ANNValueFunction - {self.shifts_trained}"
