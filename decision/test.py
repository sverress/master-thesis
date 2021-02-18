import unittest

from clustering.scripts import get_initial_state
from decision.scripts import *


class BasicDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_state = get_initial_state()

    def test_number_of_actions_battery_swaps(self):
        # Modify initial state. 5 battery swaps possible.
        self.initial_state.vehicle.scooter_inventory = []
        self.initial_state.current_cluster.scooters = self.initial_state.current_cluster.scooters[
            :5
        ]
        self.initial_state.clusters = self.initial_state.clusters[1:2]
        self.initial_state.current_cluster.ideal_state = 5

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        self.assertEqual(len(actions), 6)

    def test_number_of_actions_pick_ups(self):
        # Modify initial state. 5 battery swaps and 2 pick ups possible
        self.initial_state.vehicle.scooter_inventory = []
        self.initial_state.current_cluster.scooters = self.initial_state.current_cluster.scooters[
            :5
        ]
        self.initial_state.clusters = self.initial_state.clusters[1:2]
        self.initial_state.current_cluster.ideal_state = 3

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        self.assertEqual(len(actions), 15)

    def test_number_of_actions_deliveries(self):
        # Modify initial state. 5 battery swaps and 2 drop-offs possible
        self.initial_state.vehicle.scooter_inventory = self.initial_state.current_cluster.scooters[
            7:9
        ]
        self.initial_state.current_cluster.scooters = self.initial_state.current_cluster.scooters[
            :5
        ]
        self.initial_state.clusters = self.initial_state.clusters[1:2]
        self.initial_state.current_cluster.ideal_state = 7

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        self.assertEqual(len(actions), 18)

    def test_number_of_actions_clusters(self):
        # Modify initial state. 5 battery swaps and 2 drop-offs possible
        self.initial_state.vehicle.scooter_inventory = []
        self.initial_state.current_cluster.scooters = []
        self.initial_state.clusters = self.initial_state.clusters[1:6]
        self.initial_state.current_cluster.ideal_state = 5

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        self.assertEqual(len(actions), 5)


if __name__ == "__main__":
    unittest.main()
