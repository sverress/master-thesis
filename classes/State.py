from classes.Cluster import Cluster
from classes.Vehicle import Vehicle


class State:
    def __init__(self, clusters: [Cluster], current: Cluster, vehicle: Vehicle):
        # Have we checked that current Cluster is not included in clusters list?
        self.clusters = clusters
        self.current = current
        self.vehicle = vehicle

