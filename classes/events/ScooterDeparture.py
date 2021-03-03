from classes import Scooter
from classes.events import Event


class ScooterDeparture(Event):
    def __init__(
        self, departure_time: int, departure_cluster_id: int, scooter: Scooter
    ):
        super().__init__(departure_time)
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

        # set time of world to this event's time
        world.time = self.time
