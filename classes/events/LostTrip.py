from classes import Event
from globals import LOST_TRIP_REWARD


class LostTrip(Event):
    def __init__(self, time: int, location_id: int):
        super().__init__(time)
        self.location_id = location_id

    def perform(self, world, **kwargs) -> None:
        world.add_reward(LOST_TRIP_REWARD, self.location_id)
        super(LostTrip, self).perform(world, **kwargs)
