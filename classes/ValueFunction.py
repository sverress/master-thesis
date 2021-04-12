import classes
import numpy as np


class ValueFunction:
    def __init__(
        self,
        number_of_clusters: int,
        step_size=0.1,
        discount_factor=0.8,
        number_of_features_per_cluster=3,
    ):
        self.weights = [1 / 10] * (number_of_clusters * number_of_features_per_cluster)
        self.step_size = step_size
        self.discount_factor = discount_factor

    def estimate_value(
        self, state: classes.State, vehicle: classes.Vehicle, action: classes.Action
    ):
        current_state_features = self.convert_state_to_features(state, vehicle)

        current_state_value = float(np.dot(current_state_features, self.weights))

        reward = state.do_action(action, vehicle)

        next_state_features = self.convert_state_to_features(state, vehicle)

        next_state_value = float(np.dot(next_state_features, self.weights))

        self.update_weights(
            current_state_features, current_state_value, next_state_value, reward
        )

        return current_state_value

    def update_weights(
        self,
        current_state_features: [float],
        current_state_value: float,
        next_state_value: float,
        reward: float,
    ):
        self.weights += (
            self.step_size
            * (reward - (self.discount_factor * next_state_value) - current_state_value)
            * current_state_features
        )

    @staticmethod
    def convert_state_to_features(state: classes.State, vehicle: classes.Vehicle):
        normalized_distance = ValueFunction.normalize_list(
            state.get_distance_to_all(vehicle.current_location.id)
        )
        normalized_deviation_ideal_state = ValueFunction.normalize_list(
            [
                abs(len(cluster.scooters) - cluster.ideal_state)
                for cluster in state.clusters
            ]
        )
        normalized_deficient_battery = ValueFunction.normalize_list(
            [
                len(cluster.scooters) - cluster.get_current_state()
                for cluster in state.clusters
            ]
        )

        return (
            normalized_distance
            + normalized_deviation_ideal_state
            + normalized_deficient_battery
        )

    @staticmethod
    def normalize_list(parameter_list: [float]):
        min_value = min(parameter_list)
        max_value = max(parameter_list)

        return [
            (parameter - min_value) / (max_value - min_value)
            for parameter in parameter_list
        ]
