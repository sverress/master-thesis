from shapely.geometry import MultiPoint
from classes import Scooter
from classes.Location import Location
from globals import CLUSTER_CENTER_DELTA
import numpy as np


class Cluster(Location):
    def __init__(self, cluster_id: int, scooters: [Scooter]):
        self.id = cluster_id
        self.scooters = scooters
        self.ideal_state = 10
        self.trip_intensity_per_iteration = 10
        super().__init__(*self.__compute_center())

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
            list(map(lambda scooter: (scooter.get_location()), self.scooters))
        ).centroid
        return cluster_centroid.x, cluster_centroid.y

    def add_scooter(self, scooter: Scooter):
        # Adding scooter to scooter list
        self.scooters.append(scooter)
        # Changing coordinates of scooter to this location + some delta
        delta_lat = np.random.uniform(-CLUSTER_CENTER_DELTA, CLUSTER_CENTER_DELTA)
        delta_lon = np.random.uniform(-CLUSTER_CENTER_DELTA, CLUSTER_CENTER_DELTA)
        scooter.set_coordinates(self.get_lat() + delta_lat, self.get_lon() + delta_lon)

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

    def print_all_scooters(self, with_coordinates=False):
        string = ""
        for scooter in self.scooters:
            string += f"ID: {scooter.id}  Battery {round(scooter.battery, 1)}"
            string += (
                f"Coord: {scooter.get_location()} | " if with_coordinates else " | "
            )
        return string if string != "" else "Empty cluster"

    def get_swappable_scooters(self):
        scooters = [scooter for scooter in self.scooters if scooter.battery < 100]
        return sorted(scooters, key=lambda scooter: scooter.battery, reverse=False)

    def get_scooter_from_id(self, scooter_id):
        if scooter_id not in map(lambda scooter: scooter.id, self.scooters):
            raise ValueError(f"Scooter {scooter_id} is not in cluster {self.id}.")
        return next(
            (scooter for scooter in self.scooters if scooter.id == scooter_id), None
        )

    def __repr__(self):
        return (
            f"Cluster {self.id}: {len(self.scooters)} scooters, current state: {self.get_current_state()},"
            f" ideal state: {self.ideal_state}"
        )
