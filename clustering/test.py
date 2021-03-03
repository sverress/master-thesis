import unittest
from clustering.scripts import get_initial_state


class ClusteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = get_initial_state(500)

    def test_sample_size(self):
        self.assertEqual(len(self.state.get_scooters()), 500)

    @staticmethod
    def test_full_dataset_init_state():
        get_initial_state()

    def test_ideal_state_sum_to_number_of_scooters(self):
        self.assertAlmostEqual(
            len(self.state.get_scooters()),
            sum(cluster.ideal_state for cluster in self.state.clusters),
            -1,
        )


if __name__ == "__main__":
    unittest.main()
