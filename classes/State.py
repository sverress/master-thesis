from classes.Cluster import Cluster
from classes.Vehicle import Vehicle


class State:
    def __init__(self, clusters: [Cluster], current: Cluster, vehicle: Vehicle):
        self.clusters = clusters
        self.current = current
        self.vehicle = vehicle

