import unittest

from clustering.scripts import get_initial_state


class StateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_state = get_initial_state(500)

    def test_visualize_clusters(self):
        self.initial_state.visualize_clustering()


if __name__ == "__main__":
    unittest.main()
