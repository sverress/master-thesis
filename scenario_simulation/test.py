import unittest
from clustering.scripts import get_initial_state


class BasicScenarioSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = get_initial_state()


if __name__ == "__main__":
    unittest.main()
