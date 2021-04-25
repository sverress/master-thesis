from .abstract import *
from tensorflow import keras
import numpy as np


class ANNValueFunction(ValueFunction):
    def __init__(self, network_structure: [int], **kwargs):
        super().__init__(**kwargs)
        self.network_structure = network_structure
        self.model = keras.Sequential()

    def setup(self, state: classes.State):
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
        self,
        state: classes.State,
        vehicle: classes.Vehicle,
        time: int,
        state_features=None,
    ):
        if not state_features:
            state_features = self.get_state_features(state, vehicle, time)

        return float(self.model(np.array([state_features]))[0][0])

    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):
        self.model.fit(
            np.array([current_state_features]),
            np.array([self.discount_factor * next_state_value + reward]),
            epochs=50,
            verbose=False,
        )

    def get_state_features(
        self, state: classes.State, vehicle: classes.Vehicle, time: int
    ):
        return self.convert_state_to_features(state, vehicle, time)

    def __getstate__(self):
        """
        Method used to pickle value function.
        Not able to do it out of the box due to keras model.
        :return:
        """
        state = self.__dict__.copy()
        state["model"] = self.model.to_json()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = keras.models.model_from_json(state["model"])
