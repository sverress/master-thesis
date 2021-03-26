import unittest
import random
from classes import World, Event


class WorldTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(40, number_of_clusters=5, visualize=False)

    def test_run(self):
        self.world.stack = [Event(time) for time in range(10, 41, 10)]
        self.world.run()

    def test_add_event(self):
        # Clear initial stack
        self.world.stack = []
        shuffled_events = [Event(time) for time in range(10, 41, 10)]
        random.shuffle(shuffled_events)
        for event in shuffled_events:
            self.world.add_event(event)
        self.assertSequenceEqual(
            [event.time for event in self.world.stack], range(10, 41, 10)
        )

    def test_run_with_initial_stack(self):
        self.world.run()


if __name__ == "__main__":
    unittest.main()
