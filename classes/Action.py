from classes.Scooter import Scooter
from classes.Cluster import Cluster


class Action:
    def __init__(
        self,
        battery_swaps: [Scooter],
        pick_ups: [Scooter],
        delivery_scooters: [Scooter],
        next_cluster: Cluster,
    ):
        self.battery_swaps = battery_swaps
        self.pick_ups = pick_ups
        self.delivery_scooters = delivery_scooters
        self.next_cluster = next_cluster

    def get_action_time(self, distance: int):
        return 10
