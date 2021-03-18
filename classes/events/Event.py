class Event:
    def __init__(self, time: int):
        self.time = time

    def perform(self, world) -> None:
        if world.time <= self.time:
            world.time = self.time
        else:
            raise ValueError(
                f"{self.__class__.__name__} object tries to move the world backwards in time. Event time: {self.time}"
                f", World time: {world.time}"
            )

    @staticmethod
    def add_metric(world, time):
        world.metrics.add_analysis_metrics(world.rewards, world.state.clusters, time)
