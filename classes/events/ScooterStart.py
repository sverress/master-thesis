from classes.events import Event
from classes import Scooter


class ScooterStart(Event):
    def __int__(self, time: int, scooter: Scooter, departure_cluster_id: int):
        super().__init__(time)
        self.scooter = scooter
        self.departure_cluster_id = departure_cluster_id

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        # get departure cluster
        departure_cluster = world.state.get_cluster_by_id(self.departure_cluster_id)

        # remove scooter from the departure cluster
        departure_cluster.remove_scooter(self.scooter)
