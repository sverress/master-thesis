from classes.events import Event
from classes import World
from globals import LOST_TRIP_REWARD


class LostTrip(Event):
    def __init__(self, time):
        super().__init__(time)
        self.time = time

    def perform(self, world: World) -> None:
        world.add_reward(LOST_TRIP_REWARD)
        super(LostTrip, self).perform(world)
