import itertools
import numpy as np

from .abstract import *


class LinearValueFunction(ValueFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = None

    def setup(self, state):
        (
            number_of_locations_indicators,
            number_of_state_features,
        ) = self.get_number_of_location_indicators_and_state_features(state)
        self.weights = [self.weight_init_value] * (
            number_of_locations_indicators
            + number_of_state_features
            + (number_of_locations_indicators * number_of_state_features)
        )
        self.location_indicator = [0] * number_of_locations_indicators
        super(LinearValueFunction, self).setup(state)

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

    def get_state_features(self, state, vehicle, time):
        return self.create_location_features_combination(
            self.convert_state_to_features(state, vehicle, time)
        )
