import unittest

from classes import World, Action
from clustering.scripts import get_initial_state
from decision.policies import RandomRolloutPolicy


class BasicDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_state = get_initial_state(sample_size=100, number_of_clusters=2)

    def test_battery_swaps(self):
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
        start_battery_percentage = current_cluster.get_current_state() * 100

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        # Test number of swaps less or equal to ideal state
        for action in actions:
            self.assertLessEqual(
                len(action.battery_swaps),
                self.initial_state.current_cluster.ideal_state,
            )

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
            current_cluster.get_current_state() * 100,
            start_battery_percentage + len(actions[-1].battery_swaps) * 20.0,
        )

    def test_pick_ups(self):
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
        start_battery_percentage = current_cluster.get_current_state() * 100

        # Get all possible actions
        actions = self.initial_state.get_possible_actions()

        # Test number of actions
        self.assertEqual(len(actions), 12)

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
        self.assertAlmostEqual(
            current_cluster.get_current_state() * 100,
            start_battery_percentage
            + len(actions[-1].battery_swaps) * 80.0
            - len(actions[-1].pick_ups) * 20.0,
        )

    def test_deliveries(self):
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
        start_battery_percentage = current_cluster.get_current_state() * 100

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
        self.assertAlmostEqual(
            current_cluster.get_current_state() * 100,
            start_battery_percentage
            + len(actions[-1].battery_swaps) * 20.0
            + delivery_scootery_battery,
        )

    def test_number_of_actions_clusters(self):
        initial_state = get_initial_state(sample_size=100, number_of_clusters=6)
        # Modify initial state. 5 battery swaps and 2 drop-offs possible
        initial_state.vehicle.scooter_inventory = []
        initial_state.current_cluster.scooters = []

        # Get all possible actions
        actions = initial_state.get_possible_actions(number_of_neighbours=5)

        # Test number of actions possible
        self.assertEqual(len(actions), 5)

    def test_number_of_actions(self):
        self.assertLess(
            len(self.initial_state.get_possible_actions(divide=2)),
            len(self.initial_state.get_possible_actions()),
        )
        self.assertLess(
            len(self.initial_state.get_possible_actions(divide=2)),
            len(self.initial_state.get_possible_actions(divide=3)),
        )

    def test_random_rollout_policy(self):
        self.assertIsInstance(RandomRolloutPolicy.get_best_action(World(40)), Action)


if __name__ == "__main__":
    unittest.main()
