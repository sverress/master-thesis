from classes.events import Event


class GenerateScooterTrips(Event):
    def __init__(self, time: int):
        super().__init__(time)

    def perform(self, world) -> None:
        pass
