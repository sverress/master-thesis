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

    def get_action_time(self, distance):
        """
        Get the time consumed from performing an action (travel from cluster 1 to 2) in a given state.
        Can add time for performing actions on scooters as well.
        :param distance: distance in km from current cluster to next cluster
        :return: Total time to perform action in minutes
        """
        duration = (
            len(self.battery_swaps) + len(self.pick_ups) + len(self.delivery_scooters)
        ) * 2
        return duration + distance * 30 * 60
