import unittest
from decision.scripts import *


class BasicDecisionTests(unittest.TestCase):
    def test_number_of_actions_battery_swaps(self):
        self.initial_state = get_initial_state(sample_size=100, number_of_clusters=2)

        # Modify initial state. 5 battery swaps possible.
        self.initial_state.vehicle.scooter_inventory = []
        self.initial_state.current_cluster.scooters = self.initial_state.current_cluster.scooters[
            :5
        ]
        self.initial_state.current_cluster.ideal_state = 5

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        self.assertEqual(len(actions), 6)

    def test_number_of_actions_pick_ups(self):
        self.initial_state = get_initial_state(sample_size=100, number_of_clusters=2)

        # Modify initial state. 5 battery swaps and 2 pick ups possible
        self.initial_state.vehicle.scooter_inventory = []
        self.initial_state.current_cluster.scooters = self.initial_state.current_cluster.scooters[
            :5
        ]
        self.initial_state.current_cluster.ideal_state = 3

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        self.assertEqual(len(actions), 15)

    def test_number_of_actions_deliveries(self):
        self.initial_state = get_initial_state(sample_size=100, number_of_clusters=2)
        # Modify initial state. 5 battery swaps and 2 drop-offs possible
        self.initial_state.vehicle.scooter_inventory = self.initial_state.current_cluster.scooters[
            7:9
        ]
        self.initial_state.current_cluster.scooters = self.initial_state.current_cluster.scooters[
            :5
        ]
        self.initial_state.current_cluster.ideal_state = 7

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        self.assertEqual(len(actions), 18)

    def test_number_of_actions_clusters(self):
        self.initial_state = get_initial_state(sample_size=100, number_of_clusters=6)
        # Modify initial state. 5 battery swaps and 2 drop-offs possible
        self.initial_state.vehicle.scooter_inventory = []
        self.initial_state.current_cluster.scooters = []

        # Get all possible actions
        actions = self.initial_state.get_possible_actions(number_of_neighbours=5)

        self.assertEqual(len(actions), 5)


if __name__ == "__main__":
    unittest.main()
