import tempfile

from .abstract import *
from tensorflow import keras
import numpy as np


class ANNValueFunction(ValueFunction):
    def __init__(
        self,
        weight_update_step_size,
        weight_init_value,
        discount_factor,
        vehicle_inventory_step_size,
        location_repetition,
        network_structure: [int],
    ):
        super().__init__(
            weight_update_step_size,
            weight_init_value,
            discount_factor,
            vehicle_inventory_step_size,
            location_repetition,
        )
        self.network_structure = network_structure
        self.model = keras.Sequential()

    def setup(self, state: classes.State):
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
            self.model.add(keras.layers.Dense(layer, activation="relu"))
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
        return float(self.model(np.array([state_features]))[0][0])

    def batch_update_weights(self, batch: [(float, float, float, [float])]):
        targets = [
            self.compute_and_record_td_error(
                current_state_value, next_state_value, reward
            )
            + current_state_value
            for current_state_value, next_state_value, reward, _ in batch
        ]
        state_features = [state_feature for _, _, _, state_feature in batch]
        self.model.fit(
            np.array(state_features), np.array(targets), verbose=False,
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
            epochs=10,
            verbose=False,
        )

    def get_next_state_features(
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        action: classes.Action,
        time: int,
    ):
        return self.convert_next_state_features(state, vehicle, action, time)

    def get_state_features(
        self, state: classes.State, vehicle: classes.Vehicle, time: int
    ):
        return self.convert_state_to_features(state, vehicle, time)

    def __getstate__(self):
        state = self.__dict__.copy()
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            keras.models.save_model(self.model, fd.name, overwrite=True)
            model_str = fd.read()
        state["model"] = model_str
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(state["model"])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.model = model

    def __str__(self):
        return f"ANNValueFunction - {self.shifts_trained}"
