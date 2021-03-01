import unittest
from clustering.scripts import get_initial_state


class ClusteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = get_initial_state(500)

    def test_sample_size(self):
        self.assertEqual(len(self.state.get_scooters()), 500)


if __name__ == "__main__":
    unittest.main()
