from classes import Event
from globals import LOST_TRIP_REWARD


class LostTrip(Event):
    def __init__(self, time: int):
        super().__init__(time)

    def perform(self, world, add_metric=True) -> None:
        world.add_reward(LOST_TRIP_REWARD)
        super(LostTrip, self).perform(world, add_metric)
