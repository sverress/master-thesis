from classes import World


class Event:
    def __init__(self, time: int):
        self.time = time

    def perform(self, world: World) -> None:
        world.time = self.time
