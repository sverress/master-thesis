import unittest

import clustering.scripts
import decision
from classes import World
from scenario_simulation.scripts import estimate_reward


class BasicScenarioSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(
            100,
            decision.RandomActionPolicy(),
            clustering.scripts.get_initial_state(100, 10),
        )

    def test_estimate_reward(self):
        estimate_reward(
            self.world,
            self.world.state.vehicles[1],
            policy=decision.RandomActionPolicy(),
        )


if __name__ == "__main__":
    unittest.main()
