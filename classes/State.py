from classes.Cluster import Cluster
from classes.Vehicle import Vehicle
from math import sqrt, pi, sin, cos, atan2


class State:
    def __init__(self, clusters: [Cluster], current: Cluster, vehicle: Vehicle):
        self.clusters = clusters
        self.current = current
        self.vehicle = vehicle
        self.distance_matrix = self.calculate_distance_matrix()

    def calculate_distance_matrix(self):
        """
        Computes distance matrix for all clusters
        :return: Distance matrix
        """
        distance_matrix = []
        for cluster in self.clusters:
            neighbour_distance = []
            for neighbour in self.clusters:
                if cluster == neighbour:
                    neighbour_distance.append(0.0)
                else:
                    c_center = cluster.center
                    n_center = neighbour.center
                    neighbour_distance.append(
                        self.haversin(
                            c_center[0], c_center[1], n_center[0], n_center[1]
                        )
                    )
            distance_matrix.append(neighbour_distance)
        return distance_matrix

    @staticmethod
    def haversin(lat1, lon1, lat2, lon2):
        """
        Compute the distance between two points in meters
        :param lat1: Coordinate 1 lat
        :param lon1: Coordinate 1 lon
        :param lat2: Coordinate 2 lat
        :param lon2: Coordinate 2 lon
        :return: Meters between coordinates
        """
        radius = 6378.137
        d_lat = lat2 * pi / 180 - lat1 * pi / 180
        d_lon = lon2 * pi / 180 - lon1 * pi / 180
        a = sin(d_lat / 2) * sin(d_lat / 2) + cos(lat1 * pi / 180) * cos(
            lat2 * pi / 180
        ) * sin(d_lon / 2) * sin(d_lon / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = radius * c
        return distance
