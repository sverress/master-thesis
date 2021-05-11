import tempfile

from .abstract import *
from tensorflow import keras
import tensorflow as tf
import numpy as np
from decision.value_functions.helpers import SplitGD


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
        self.model = SplitGD(trace_decay=trace_decay, discount_factor=discount_factor)

    def setup(self, state: classes.State):
        from tensorflow import keras

        if self.setup_complete:
            return
        (
            number_of_locations_indicators,
            number_of_state_features,
        ) = self.get_number_of_location_indicators_and_state_features(state)
        # First layer needs input size as argument
        nodes_in_first_layer, *nodes_in_rest_of_layers = self.network_structure
        self.model.add(
            keras.layers.Dense(
                nodes_in_first_layer,
                input_dim=(number_of_locations_indicators + number_of_state_features),
            )
        )
        for layer in nodes_in_rest_of_layers:
            self.model.add(keras.layers.Dropout(0.5))
            self.model.add(
                keras.layers.Dense(
                    layer, activation="relu", kernel_regularizer=keras.regularizers.L2()
                )
            )
        # The last layer needs to have a single value function output
        self.model.add(keras.layers.Dense(1))
        self.model.compile(
            loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]
        )
        super(ANNValueFunction, self).setup(state)

    def estimate_value(
        self, state: classes.State, vehicle: classes.Vehicle, time: int,
    ):

        return self.estimate_value_from_state_features(
            self.get_state_features(state, vehicle, time)
        )

    def estimate_value_from_state_features(self, state_features: [float]):
        return float(self.model(tf.convert_to_tensor([state_features]))[0][0])

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

        self.model.fit(
            np.array(state_features), np.array(targets), td_errors, verbose=False,
        )

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
            verbose=False,
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
        from tensorflow import keras

        state = self.__dict__.copy()
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            keras.models.save_model(self.model, fd.name, overwrite=True)
            model_str = fd.read()
        state["model"] = model_str
        return state

    def __setstate__(self, state):
        from tensorflow import keras

        self.__dict__.update(state)
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(state["model"])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.model = model

    def __str__(self):
        return f"ANNValueFunction - {self.shifts_trained}"
