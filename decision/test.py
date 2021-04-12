import unittest
import random

from classes import World, Action, Scooter
from clustering.scripts import get_initial_state
from decision.policies import RandomRolloutPolicy, SwapAllPolicy
from decision.neighbour_filtering import filtering_neighbours
import decision.helpers


class BasicDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_state = get_initial_state(
            sample_size=100, number_of_clusters=2, initial_location_depot=False
        )
        self.vehicle = self.initial_state.vehicles[0]

    def test_battery_swaps(self):
        # Modify initial state. 5 battery swaps possible.
        self.vehicle.scooter_inventory = []
        # Let the current location of the vehicle contain 5 scooters
        self.vehicle.current_location.scooters = self.vehicle.current_location.scooters[
            :5
        ]
        self.vehicle.current_location.ideal_state = 5
        start_number_of_scooters = len(self.vehicle.current_location.scooters)
        current_cluster = self.vehicle.current_location

        for scooter in self.vehicle.current_location.scooters:
            scooter.battery = 80.0
        start_battery_percentage = current_cluster.get_current_state() * 100

        # Get all possible actions
        actions = self.initial_state.get_possible_actions(
            self.vehicle, number_of_neighbours=6
        )

        # Test number of swaps less or equal to ideal state
        for action in actions:
            self.assertLessEqual(
                len(action.battery_swaps), self.vehicle.current_location.ideal_state,
            )

        # Test number of actions
        self.assertEqual(len(actions), 6)

        # Calculate the expected reward
        reward = (
            len(actions[-1].battery_swaps)
            * 0.2
            * self.vehicle.current_location.prob_of_scooter_usage()
        )

        # Test reward
        self.assertEqual(
            self.initial_state.do_action(actions[-1], self.vehicle), reward
        )

        # Test number of scooters
        self.assertEqual(len(current_cluster.scooters), start_number_of_scooters)

        # Test battery percentage
        self.assertEqual(
            current_cluster.get_current_state() * 100,
            start_battery_percentage + len(actions[-1].battery_swaps) * 20.0,
        )

    def test_pick_ups(self):
        # Modify initial state. 5 battery swaps and 2 pick ups possible
        self.vehicle.scooter_inventory = []
        self.vehicle.current_location.scooters = self.vehicle.current_location.scooters[
            :5
        ]
        self.vehicle.current_location.ideal_state = 3
        start_number_of_scooters = len(self.vehicle.current_location.scooters)
        current_cluster = self.vehicle.current_location

        # Set all battery to 20% to calculate expected reward
        for scooter in self.vehicle.current_location.scooters:
            scooter.battery = 20.0
        start_battery_percentage = current_cluster.get_current_state() * 100

        # Get all possible actions
        actions = self.initial_state.get_possible_actions(self.vehicle)

        # Test number of actions
        self.assertEqual(len(actions), 15)

        # Test no reward for pickup
        self.assertEqual(
            round(self.initial_state.do_action(actions[-1], self.vehicle), 1), 0
        )

        # Test number of scooters
        self.assertEqual(
            len(current_cluster.scooters),
            start_number_of_scooters - len(actions[-1].pick_ups),
        )

        # Test inventory vehicle
        self.assertEqual(
            start_number_of_scooters - len(current_cluster.scooters),
            len(self.vehicle.scooter_inventory),
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
        self.vehicle.scooter_inventory = self.vehicle.current_location.scooters[7:9]
        self.vehicle.current_location.scooters = self.vehicle.current_location.scooters[
            :5
        ]
        self.vehicle.current_location.ideal_state = 7
        start_number_of_scooters = len(self.vehicle.current_location.scooters)
        current_cluster = self.vehicle.current_location

        # Set all battery to 80% to calculate expected reward
        for scooter in self.vehicle.current_location.scooters:
            scooter.battery = 80.0
        start_battery_percentage = current_cluster.get_current_state() * 100

        # Get all possible actions
        actions = self.initial_state.get_possible_actions(
            self.vehicle, number_of_neighbours=6
        )

        # Test number of actions
        self.assertEqual(len(actions), 18)

        # Calculate the expected reward
        reward = (
            len(actions[-1].battery_swaps)
            * 0.2
            * self.vehicle.current_location.prob_of_scooter_usage()
            + len(actions[-1].delivery_scooters) * 1.0
        )

        # Test reward
        self.assertEqual(
            self.initial_state.do_action(actions[-1], self.vehicle), reward
        )

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
        delivery_scooter_battery = sum(
            map(
                lambda delivery_scooter: delivery_scooter.battery,
                delivery_scooter_objects,
            )
        )
        self.assertAlmostEqual(
            current_cluster.get_current_state() * 100,
            start_battery_percentage
            + len(actions[-1].battery_swaps) * 20.0
            + delivery_scooter_battery,
        )

    def test_number_of_actions_clusters(self):
        initial_state = get_initial_state(
            sample_size=100, number_of_clusters=6, initial_location_depot=False
        )
        vehicle = initial_state.vehicles[0]
        # Modify initial state. 5 battery swaps and 2 drop-offs possible
        vehicle.scooter_inventory = []
        vehicle.current_location.scooters = []

        # Get all possible actions
        actions = initial_state.get_possible_actions(vehicle, number_of_neighbours=5)

        # Test number of actions possible
        self.assertEqual(5, len(actions))

    def test_number_of_actions(self):
        bigger_state = get_initial_state(sample_size=1000, initial_location_depot=False)
        bigger_state.current_location = random.choice(
            [
                cluster
                for cluster in bigger_state.clusters
                if cluster.number_of_scooters() > 0
            ]
        )
        self.assertLess(
            len(bigger_state.get_possible_actions(self.vehicle, divide=2)),
            len(bigger_state.get_possible_actions(self.vehicle)),
        )
        self.assertLess(
            len(bigger_state.get_possible_actions(self.vehicle, divide=2)),
            len(bigger_state.get_possible_actions(self.vehicle, divide=4)),
        )


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(40)
        self.vehicle = self.world.state.vehicles[0]

    def test_random_rollout_policy(self):
        self.assertIsInstance(
            RandomRolloutPolicy.get_best_action(self.world, self.vehicle), Action,
        )

    def test_swap_all_policy(self):
        action = SwapAllPolicy.get_best_action(self.world, self.vehicle)
        self.assertIsInstance(action, Action)
        self.assertEqual(len(action.pick_ups), 0)
        self.assertEqual(len(action.delivery_scooters), 0)

    def test_epsilon_greedy_function(self):
        choices = [(1.3, "Hello"), (214, "Hei"), (0.01, "Nix")]
        self.assertIsInstance(decision.helpers.epsilon_greedy(choices), str)
        self.assertEqual(decision.helpers.epsilon_greedy(choices, 0.0), "Hei")


class NeighbourFilteringTests(unittest.TestCase):
    def test_filtering_neighbours(self):
        state = get_initial_state(100, 10)
        vehicle = state.vehicles[0]

        best_neighbours_with_random = filtering_neighbours(
            state, vehicle, number_of_neighbours=3, number_of_random_neighbours=1,
        )

        # test if the number of neighbours is the same, even though one is random
        self.assertEqual(len(best_neighbours_with_random), 3)

        sorted_neighbours = state.get_neighbours(
            vehicle.current_location, is_sorted=True
        )[:3]
        for cluster in state.clusters:
            if cluster in sorted_neighbours:
                cluster.ideal_state = 100
                for scooter in cluster.scooters:
                    scooter.battery = 0

        best_neighbours = filtering_neighbours(state, vehicle, number_of_neighbours=3)

        # add one scooter to vehicle inventory so filtering neighbours uses the right filtering method
        vehicle.pick_up(Scooter(0, 0, 0.9, 0))

        # check if clusters are closest and with the highest deviation -> best neighbours
        for neighbour in best_neighbours:
            self.assertTrue(neighbour in sorted_neighbours)


if __name__ == "__main__":
    unittest.main()
