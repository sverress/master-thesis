import unittest
from decision.scripts import *


class BasicDecisionTests(unittest.TestCase):
    def test_battery_swaps(self):
        self.initial_state = get_initial_state(sample_size=100, number_of_clusters=2)

        # Modify initial state. 5 battery swaps possible.
        self.initial_state.vehicle.scooter_inventory = []
        self.initial_state.current_cluster.scooters = self.initial_state.current_cluster.scooters[
            :5
        ]
        self.initial_state.current_cluster.ideal_state = 5
        start_number_of_scooters = len(self.initial_state.current_cluster.scooters)
        current_cluster = self.initial_state.current_cluster

        for scooter in self.initial_state.current_cluster.scooters:
            scooter.battery = 80.0
        start_battery_percentage = current_cluster.get_current_state()

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        # Test number of actions
        self.assertEqual(len(actions), 6)

        # Calculate the expected reward
        reward = len(actions[-1].battery_swaps) * 0.2

        # Test reward
        self.assertEqual(self.initial_state.do_action(actions[-1]), reward)

        # Test number of scooters
        self.assertEqual(len(current_cluster.scooters), start_number_of_scooters)

        # Test battery percentage
        self.assertEqual(
            current_cluster.get_current_state(),
            start_battery_percentage + len(actions[-1].battery_swaps) * 20.0,
        )

    def test_pick_ups(self):
        self.initial_state = get_initial_state(sample_size=100, number_of_clusters=2)

        # Modify initial state. 5 battery swaps and 2 pick ups possible
        self.initial_state.vehicle.scooter_inventory = []
        self.initial_state.current_cluster.scooters = self.initial_state.current_cluster.scooters[
            :5
        ]
        self.initial_state.current_cluster.ideal_state = 3
        start_number_of_scooters = len(self.initial_state.current_cluster.scooters)
        current_cluster = self.initial_state.current_cluster

        # Set all battery to 20% to calculate expected reward
        for scooter in self.initial_state.current_cluster.scooters:
            scooter.battery = 20.0
        start_battery_percentage = current_cluster.get_current_state()

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        # Test number of actions
        self.assertEqual(len(actions), 15)

        # Calculate the expected reward
        reward = len(actions[-1].battery_swaps) * 0.8 - len(actions[-1].pick_ups) * 0.2

        # Test reward
        self.assertEqual(
            round(self.initial_state.do_action(actions[-1]), 1), round(reward, 1)
        )

        # Test number of scooters
        self.assertEqual(
            len(current_cluster.scooters),
            start_number_of_scooters - len(actions[-1].pick_ups),
        )

        # Test inventory vehicle
        self.assertEqual(
            start_number_of_scooters - len(current_cluster.scooters),
            len(self.initial_state.vehicle.scooter_inventory),
        )

        # Test battery percentage
        self.assertEqual(
            current_cluster.get_current_state(),
            start_battery_percentage
            + len(actions[-1].battery_swaps) * 80.0
            - len(actions[-1].pick_ups) * 20.0,
        )

    def test_deliveries(self):
        self.initial_state = get_initial_state(sample_size=100, number_of_clusters=2)
        # Modify initial state. 5 battery swaps and 2 drop-offs possible
        self.initial_state.vehicle.scooter_inventory = self.initial_state.current_cluster.scooters[
            7:9
        ]
        self.initial_state.current_cluster.scooters = self.initial_state.current_cluster.scooters[
            :5
        ]
        self.initial_state.current_cluster.ideal_state = 7
        start_number_of_scooters = len(self.initial_state.current_cluster.scooters)
        current_cluster = self.initial_state.current_cluster

        # Set all battery to 80% to calculate expected reward
        for scooter in self.initial_state.current_cluster.scooters:
            scooter.battery = 80.0
        start_battery_percentage = current_cluster.get_current_state()

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        # Test number of actions
        self.assertEqual(len(actions), 18)

        # Calculate the expected reward
        reward = (
            len(actions[-1].battery_swaps) * 0.2
            + len(actions[-1].delivery_scooters) * 1.0
        )

        # Test reward
        self.assertEqual(self.initial_state.do_action(actions[-1]), round(reward, 1))

        # Test number of scooters
        self.assertEqual(
            len(current_cluster.scooters),
            start_number_of_scooters + len(actions[-1].delivery_scooters),
        )

        # Test battery percentage
        delivery_scooter_objects = [
            scooter
            for scooter in current_cluster.scooters
            if scooter.id in actions[-1].delivery_scooters
        ]
        delivery_scootery_battery = sum(
            map(lambda scooter: scooter.battery, delivery_scooter_objects)
        )
        self.assertEqual(
            current_cluster.get_current_state(),
            start_battery_percentage
            + len(actions[-1].battery_swaps) * 20.0
            + delivery_scootery_battery,
        )

    def test_number_of_actions_clusters(self):
        self.initial_state = get_initial_state(sample_size=100, number_of_clusters=6)
        # Modify initial state. 5 battery swaps and 2 drop-offs possible
        self.initial_state.vehicle.scooter_inventory = []
        self.initial_state.current_cluster.scooters = []

        # Get all possible actions
        actions = self.initial_state.get_possible_actions(number_of_neighbours=5)

        # Test number of actions possible
        self.assertEqual(len(actions), 5)


if __name__ == "__main__":
    unittest.main()
