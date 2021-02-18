from shapely.geometry import MultiPoint
import numpy as np

from classes.Scooter import Scooter


class Cluster:
    def __init__(
        self, cluster_id: int, scooters: [Scooter], movement_probabilities: np.ndarray,
    ):
        self.id = cluster_id
        self.scooters = scooters
        self.ideal_state = 2
        self.trip_intensity_per_iteration = 2
        self.center = self.__compute_center()
        self.move_probabilities = movement_probabilities

    def get_current_state(self) -> float:
        return sum(map(lambda scooter: scooter.battery, self.scooters))

    def prob_stay(self):
        return self.move_probabilities[self.id]

    def prob_leave(self, cluster):
        return self.move_probabilities[cluster.id]

    def number_of_possible_pickups(self):
        if self.number_of_scooters() > self.ideal_state:
            return 0
        else:
            return self.number_of_scooters()

    def number_of_scooters(self):
        return len(self.scooters)

    def __compute_center(self):
        cluster_centroid = MultiPoint(
            list(map(lambda scooter: (scooter.lat, scooter.lon), self.scooters))
        ).centroid
        return cluster_centroid.x, cluster_centroid.y

    def add_scooter(self, scooter: Scooter):
        self.scooters.append(scooter)

    def get_valid_scooters(self, battery_limit: float):
        return [
            scooter for scooter in self.scooters if scooter.battery >= battery_limit
        ]

    def print_all_scooters(self):
        string = ""
        for scooter in self.scooters:
            string += f"ID: {scooter.id}  Battery {round(scooter.battery, 1)} | "
        return string if string != "" else "Empty cluster"

    def __str__(self):
        return (
            f"Cluster: {len(self.scooters)} scooters, current state: {self.get_current_state()},"
            f" ideal state: {self.ideal_state}"
        )
