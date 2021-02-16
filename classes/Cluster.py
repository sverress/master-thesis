from shapely.geometry import MultiPoint

from classes.Scooter import Scooter


class Cluster:
    def __init__(self, scooters: [Scooter]):
        self.scooters = scooters
        self.current_state = self.__compute_current_state()
        self.ideal_state = 2
        self.center = self.__compute_center()

    def __compute_current_state(self):
        return sum(map(lambda scooter: scooter.battery, self.scooters))

    def dist(self, cluster):
        return 5

    def prob_stay(self):
        return 0.5

    def prob_leave(self, cluster):
        return 0.5 / 7  # 7 is number of clusters

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
