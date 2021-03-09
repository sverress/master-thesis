from classes import World


class Event:
    def __init__(self, time: int):
        self.time = time

    def perform(self, world: World) -> None:
        if world.time <= self.time:
            world.time = self.time
        else:
            raise ValueError(
                f"{self.__class__.__name__} object tries to move the world backwards in time. Event time: {self.time}"
                f", World time: {world.time}"
            )
