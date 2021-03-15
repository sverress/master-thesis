import unittest
from clustering.scripts import get_initial_state


class ClusteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state_mid = get_initial_state(500, cache=False, save=False)
        self.state_small = get_initial_state(100)
        self.state_big = get_initial_state(2000)

    def test_sample_size(self):
        self.assertEqual(len(self.state_mid.get_scooters()), 500)

    def test_ideal_state_sum_to_number_of_scooters(self):
        self.assertAlmostEqual(
            len(self.state_mid.get_scooters()),
            sum(cluster.ideal_state for cluster in self.state_mid.clusters),
            -1,
        )

    def test_move_probabilities(self):
        for state in [self.state_mid, self.state_small, self.state_mid]:
            for cluster in state.clusters:
                self.assertAlmostEqual(sum(cluster.move_probabilities), 1)
                self.assertFalse(
                    any([prob < 0 for prob in cluster.move_probabilities]),
                    "There are negative probabilities in the move probabilities matrix",
                )
                self.assertFalse(
                    any([prob > 1 for prob in cluster.move_probabilities]),
                    "There are probabilities bigger than one in the move probabilities matrix",
                )


if __name__ == "__main__":
    unittest.main()
