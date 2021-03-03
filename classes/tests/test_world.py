import unittest

from classes import World


class EventMock:
    def __init__(self, time):
        self.time = time

    def perform(self, world):
        world.time = self.time


class WorldTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(40)

    def test_run(self):
        self.world.stack = [EventMock(time) for time in range(10, 41, 10)]
        self.world.run()


if __name__ == "__main__":
    unittest.main()
