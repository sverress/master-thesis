from classes.events import Event
from classes import Scooter


class ScooterArrival(Event):
    def __int__(
        self, time: int, scooter: Scooter, arrival_cluster_id: int, distance: int
    ):
        super().__init__(time)
        self.scooter = scooter
        self.arrival_cluster_id = arrival_cluster_id
        self.distance = distance

    def perform(self, world) -> None:
        """
            :param world: world object
        """

        # get arrival cluster
        arrival_cluster = world.state.get_cluster(self.arrival_cluster_id)

        # make the scooter travel the distance to change battery
        self.scooter.travel(self.distance)

        # add scooter to the arrived cluster
        arrival_cluster.add_scooter(self.scooter)

        lat, lon = arrival_cluster.center

        # change coordinates of scooter after arrival
        self.scooter.change_coordinates(lat, lon)
