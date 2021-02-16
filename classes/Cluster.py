from shapely.geometry import MultiPoint

from classes.Scooter import Scooter


class Cluster:
    def __init__(self, scooters: [Scooter]):
        # sorting scooters after battery percent
        self.scooters = scooters
        self.ideal_state = 2
        self.trip_intensity_per_iteration = 2
        self.center = self.__compute_center()

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
            list(map(lambda scooter: (scooter.lat, scooter.lon), self.scooters))
        ).centroid
        return cluster_centroid.x, cluster_centroid.y

    def add_scooter(self, scooter: Scooter):
        self.scooters.append(scooter)

    def get_valid_scooters(self, battery_limit):
        return [s for s in self.scooters if s.battery >= battery_limit]

    def __str__(self):
        string = ""
        for s in self.scooters:
            string += f"ID: {s.id}  Battery {round(s.battery,1)} | "
        return string if string != "" else "Empty cluster"

    def get_swappable_scooters(self):
        scooters = [scooter for scooter in self.scooters if scooter.battery < 100]
        return sorted(scooters, key=lambda scooter: scooter.battery, reverse=False)
