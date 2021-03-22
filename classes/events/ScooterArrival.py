from classes import Event
import classes


class ScooterArrival(Event):
    def __init__(
        self,
        arrival_time: int,
        scooter: classes.Scooter,
        arrival_cluster_id: int,
        departure_cluster_id: int,
        distance: int,
    ):
        super().__init__(arrival_time)
        self.scooter = scooter
        self.arrival_cluster_id = arrival_cluster_id
        self.departure_cluster_id = departure_cluster_id
        self.distance = distance

    def perform(self, world, add_metric=True) -> None:
        """
            :param world: world object
        """

        # get arrival cluster
        arrival_cluster = world.state.get_cluster_by_id(self.arrival_cluster_id)

        # make the scooter travel the distance to change battery
        self.scooter.travel(self.distance)

        # add scooter to the arrived cluster (location is changed in add_scooter method)
        arrival_cluster.add_scooter(self.scooter)

        # adding the trip to world flow for visualizing purposes
        world.add_trip_to_flow(self.departure_cluster_id, self.arrival_cluster_id)

        # set time of world to this event's time
        super(ScooterArrival, self).perform(world, add_metric)
