import unittest
from clustering.scripts import get_initial_state


class ClusteringTests(unittest.TestCase):
    def test_sample_size(self):
        state = get_initial_state(500)
        self.assertEqual(len(state.get_scooters()), 500)


if __name__ == "__main__":
    unittest.main()
