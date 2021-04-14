import unittest
import decision
from classes import World
from scenario_simulation.scripts import estimate_reward


class BasicScenarioSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(100, policy=decision.SwapAllPolicy())

    def test_estimate_reward(self):
        estimate_reward(self.world, self.world.state.vehicles[0])


if __name__ == "__main__":
    unittest.main()
