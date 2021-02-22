from shapely.geometry import MultiPoint
from classes import Scooter
from classes.Location import Location
from globals import CLUSTER_CENTER_DELTA
import numpy as np


class Cluster:
    def __init__(self, cluster_id: int, scooters: [Scooter]):
        # sorting scooters after battery percent
        self.id = cluster_id
        self.scooters = scooters
        self.ideal_state = 2
        self.trip_intensity_per_iteration = 10
        self.location = self.__compute_center()

    def get_current_state(self):
        return sum(map(lambda scooter: scooter.battery, self.scooters))

    def dist(self, cluster):
        return 5

    def prob_stay(self):
        return 0.5

    def prob_leave(self, cluster):
        return 0.5 / 7  # 7 is number of clusters

    def number_of_possible_pickups(self):
        if self.number_of_scooters() <= self.ideal_state:
            return 0
        else:
            return self.number_of_scooters() - self.ideal_state

    def number_of_scooters(self):
        return len(self.scooters)

    def __compute_center(self):
        cluster_centroid = MultiPoint(
            list(
                map(
                    lambda scooter: (
                        scooter.location.get_lat(),
                        scooter.location.get_lon(),
                    ),
                    self.scooters,
                )
            )
        ).centroid
        return Location(cluster_centroid.x, cluster_centroid.y)

    def add_scooter(self, scooter: Scooter):
        # Adding scooter to scooter list
        self.scooters.append(scooter)
        # Changing coordinates of scooter to this location + some delta
        delta = np.random.uniform(-CLUSTER_CENTER_DELTA, CLUSTER_CENTER_DELTA)
        scooter.change_coordinates(
            self.location.get_lat() + delta, self.location.get_lon() + delta
        )

    def remove_scooter(self, scooter: Scooter):
        if scooter in self.scooters:
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
