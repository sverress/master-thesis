import random

from .abstract import *
from decision.value_functions.ANN import ANN


class ANNValueFunction(ValueFunction):
    def __init__(
        self,
        learning_rate,
        weight_init_value,
        discount_factor,
        vehicle_inventory_step_size,
        location_repetition,
        trace_decay,
        network_structure: [int],
    ):
        super().__init__(
            learning_rate,
            weight_init_value,
            discount_factor,
            vehicle_inventory_step_size,
            location_repetition,
            trace_decay,
        )
        self.network_structure = network_structure
        self.model = None
        self.train_count = 0

    def setup(self, state):
        if self.setup_complete:
            return
        (
            number_of_locations_indicators,
            number_of_state_features,
        ) = self.get_number_of_location_indicators_and_state_features(state)
        # One hot encoding for next location. One parameter for each sub-action (battery-swap e.g.)
        action_input_size = len(state.locations) + 3
        self.model = ANN(
            self.network_structure,
            number_of_locations_indicators
            + number_of_state_features
            + action_input_size,
            self.trace_decay,
            self.discount_factor,
            self.step_size,
        )
        super(ANNValueFunction, self).setup(state)

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        random_sample = random.sample(self.replay_buffer, 256)
        # Create training data from random sample
        states, targets = [], []
        for i, (
            sap_features,
            reward,
            next_sap_features,
        ) in enumerate(random_sample):
            states.append(sap_features)
            next_state_value = (
                self.model.predict(next_sap_features) if next_sap_features else 0
            )
            targets.append(next_state_value + reward)
        self.model.batch_fit(states, targets, verbose=1, batch_size=256, shuffle=False)
        if self.train_count % 5:
            self.model.update_predict_model()
        self.train_count += 1

    def estimate_value(
        self,
        state,
        vehicle,
        time: int,
    ):

        return self.estimate_value_from_state_features(
            self.get_state_features(state, vehicle, time)
        )

    def estimate_value_from_state_features(self, state_features: [float]):
        return self.model.predict(state_features)

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
            [current_state_features],
            [td_error + current_state_value],
            epochs=10,
        )

    def get_next_state_features(
        self,
        state,
        vehicle,
        action,
        time: int,
        cache=None,  # current_states, available_scooters = cache
    ):
        return self.convert_next_state_features(state, vehicle, action, time, cache)

    def get_state_features(self, state, vehicle, time: int, cache=None):
        return self.convert_state_to_features(state, vehicle, time, cache=cache)

    def __str__(self):
        return f"ANNValueFunction - {self.shifts_trained}"
