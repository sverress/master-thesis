import unittest
from clustering.scripts import get_initial_state
from scenario_simulation.scripts import markov_decision_process, estimate_reward


class BasicScenarioSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = get_initial_state()

    def test_estimate_reward(self):
        estimate_reward(self.state, 20)


if __name__ == "__main__":
    unittest.main()
