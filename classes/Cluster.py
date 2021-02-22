from shapely.geometry import MultiPoint
import numpy as np
from classes.Scooter import Scooter


class Cluster:
    def __init__(self, cluster_id: int, scooters: [Scooter]):
        self.id = cluster_id
        self.scooters = scooters
        self.ideal_state = 2
        self.trip_intensity_per_iteration = 10
        self.center = self.__compute_center()
        self.move_probabilities = None

    def get_current_state(self) -> float:
        return sum(map(lambda scooter: scooter.battery, self.scooters))

    def prob_stay(self):
        if self.move_probabilities:
            return self.move_probabilities[self.id]
        else:
            raise ValueError(
                "Move probabilities matrix not initialized. Please set in set_move_probabilities_function"
            )

    def prob_leave(self, cluster):
        if self.move_probabilities:
            return self.move_probabilities[cluster.id]
        else:
            raise ValueError(
                "Move probabilities matrix not initialized. Please set in set_move_probabilities_function"
            )

    def set_move_probabilities(self, move_probabilities: np.ndarray):
        self.move_probabilities = move_probabilities

    def number_of_possible_pickups(self):
        if self.number_of_scooters() <= self.ideal_state:
            return 0
        else:
            return self.number_of_scooters() - self.ideal_state

    def number_of_scooters(self):
        return len(self.scooters)

    def __compute_center(self):
        cluster_centroid = MultiPoint(
            list(map(lambda scooter: (scooter.lat, scooter.lon), self.scooters))
        ).centroid
        return cluster_centroid.x, cluster_centroid.y

    def add_scooter(self, scooter: Scooter):
        self.scooters.append(scooter)

    def remove_scooter(self, scooter: Scooter):
        if self.scooters.contains(scooter):
            self.scooters.remove(scooter)
        else:
            raise ValueError(
                "Can't remove a scooter from a cluster its not currently in"
            )

    def get_valid_scooters(self, battery_limit: float):
        return [
            scooter for scooter in self.scooters if scooter.battery >= battery_limit
        ]

    def print_all_scooters(self):
        string = ""
        for scooter in self.scooters:
            string += f"ID: {scooter.id}  Battery {round(scooter.battery, 1)} | "
        return string if string != "" else "Empty cluster"

    def get_swappable_scooters(self):
        scooters = [scooter for scooter in self.scooters if scooter.battery < 100]
        return sorted(scooters, key=lambda scooter: scooter.battery, reverse=False)

    def __repr__(self):
        return (
            f"Cluster {self.id}: {len(self.scooters)} scooters, current state: {self.get_current_state()},"
            f" ideal state: {self.ideal_state}"
        )
