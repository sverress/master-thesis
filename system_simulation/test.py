import unittest

from clustering.scripts import get_initial_state
from system_simulation.scripts import system_simulate


class BasicSystemSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_state = get_initial_state()

    def test_equal_number_of_scooters(self):
        number_of_scooters_before = sum(
            map(lambda cluster: len(cluster.scooters), self.initial_state.clusters)
        )

        after_simulation_state = system_simulate(self.initial_state)
        number_of_scooters_after = sum(
            map(lambda cluster: len(cluster.scooters), after_simulation_state.clusters)
        )
        self.assertEqual(number_of_scooters_before, number_of_scooters_after)

    def test_less_or_equal_total_battery(self):
        total_battery_of_scooters_before = sum(
            map(
                lambda cluster: sum([scooter.battery for scooter in cluster.scooters]),
                self.initial_state.clusters,
            )
        )

        after_simulation_state = system_simulate(self.initial_state)
        total_battery_of_scooters_after = sum(
            map(
                lambda cluster: sum([scooter.battery for scooter in cluster.scooters]),
                after_simulation_state.clusters,
            )
        )
        self.assertLessEqual(
            total_battery_of_scooters_after, total_battery_of_scooters_before
        )


if __name__ == "__main__":
    unittest.main()
