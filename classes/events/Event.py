from abc import ABC, abstractmethod


class Event(ABC):
    def __init__(self, time: int):
        self.time = time

    @abstractmethod
    def perform(self, world) -> None:
        pass
