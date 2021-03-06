import itertools
import numpy as np

from .abstract import *


class LinearValueFunction(ValueFunction):
    def __init__(
        self,
        weight_update_step_size,
        weight_init_value,
        discount_factor,
        vehicle_inventory_step_size,
        location_repetition,
    ):
        super().__init__(
            weight_update_step_size,
            weight_init_value,
            discount_factor,
            vehicle_inventory_step_size,
            location_repetition,
        )
        self.weights = None

    def setup(self, state):
        if self.setup_complete:
            return
        (
            number_of_locations_indicators,
            number_of_state_features,
        ) = self.get_number_of_location_indicators_and_state_features(state)
        self.weights = [self.weight_init_value] * (
            1  # Bias
            + number_of_locations_indicators
            + number_of_state_features
            + len(
                list(
                    itertools.combinations(
                        list(
                            range(
                                number_of_locations_indicators
                                + number_of_state_features
                            )
                        ),
                        2,
                    )
                )
            )
        )
        self.location_indicator = [0] * number_of_locations_indicators
        super(LinearValueFunction, self).setup(state)

    def estimate_value(
        self,
        state,
        vehicle,
        time,
    ):
        return self.estimate_value_from_state_features(
            self.get_state_features(state, vehicle, time)
        )

    def estimate_value_from_state_features(self, state_features: [float]):
        return float(np.dot(self.weights, state_features))

    def batch_update_weights(
        self, state_features, batch: [([float], float, float, float)]
    ):
        for (
            current_state_value,
            next_state_value,
            reward,
        ) in batch:
            self.update_weights(
                state_features, current_state_value, next_state_value, reward
            )

    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):

        self.weights += np.multiply(
            self.step_size
            * self.compute_and_record_td_error(
                current_state_value, next_state_value, reward
            ),
            current_state_features,
        )

    @Decorators.check_setup
    def create_location_features_combination(self, state_features):

        locations_features_combination = list(
            [
                factor1 * factor2
                for factor1, factor2 in itertools.combinations(state_features, 2)
            ]
        )

        return [1] + state_features + locations_features_combination

    def get_state_features(self, state, vehicle, time):
        return self.create_location_features_combination(
            self.convert_state_to_features(state, vehicle, time)
        )

    def __str__(self):
        return f"LinearValueFunction - {self.shifts_trained}"
