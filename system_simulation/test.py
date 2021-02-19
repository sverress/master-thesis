import unittest
from clustering.scripts import get_initial_state


class BasicSystemSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = get_initial_state()

    def test_equal_number_of_scooters(self):
        number_of_scooters_before = sum(
            map(lambda cluster: len(cluster.scooters), self.state.clusters)
        )

        self.state.system_simulate()
        number_of_scooters_after = sum(
            map(lambda cluster: len(cluster.scooters), self.state.clusters)
        )
        self.assertEqual(number_of_scooters_before, number_of_scooters_after)

    def test_less_or_equal_total_battery(self):
        total_battery_of_scooters_before = sum(
            map(
                lambda cluster: sum([scooter.battery for scooter in cluster.scooters]),
                self.state.clusters,
            )
        )

        self.state.system_simulate()
        total_battery_of_scooters_after = sum(
            map(
                lambda cluster: sum([scooter.battery for scooter in cluster.scooters]),
                self.state.clusters,
            )
        )
        self.assertLessEqual(
            total_battery_of_scooters_after, total_battery_of_scooters_before
        )


if __name__ == "__main__":
    unittest.main()
