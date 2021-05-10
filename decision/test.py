import copy
import unittest
import random

import analysis.evaluate_policies
import classes
import clustering.scripts
import decision
import decision.value_functions
import globals
from classes import World, Action, Scooter
from clustering.scripts import get_initial_state
from decision.neighbour_filtering import filtering_neighbours


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
            self.vehicle,
            number_of_neighbours=6,
        )

        # Test number of swaps less or equal to ideal state
        for action in actions:
            self.assertLessEqual(
                len(action.battery_swaps),
                self.vehicle.current_location.ideal_state,
            )

        # Test number of actions
        self.assertEqual(len(actions), 5)

        # Calculate the expected reward
        reward = (
            len(actions[-1].battery_swaps)
            * 0.2
            * self.vehicle.current_location.prob_of_scooter_usage(
                len(self.vehicle.current_location.get_available_scooters())
            )
        )

        # Test reward
        self.assertEqual(
            self.initial_state.do_action(actions[-1], self.vehicle, 0)[0], reward
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
        actions = self.initial_state.get_possible_actions(
            self.vehicle, number_of_neighbours=1
        )

        # Test number of actions
        self.assertEqual(len(actions), 14)

        # Test no reward for pickup
        self.assertEqual(
            round(self.initial_state.do_action(actions[-1], self.vehicle, 0)[0], 1), 0
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
        self.assertEqual(len(actions), 17)

        # Calculate the expected reward
        reward = (
            len(actions[-1].battery_swaps)
            * 0.2
            * self.vehicle.current_location.prob_of_scooter_usage(
                len(self.vehicle.current_location.get_available_scooters())
            )
            + len(actions[-1].delivery_scooters) * 1.0
        )

        # Test reward
        self.assertEqual(
            self.initial_state.do_action(actions[-1], self.vehicle, 0)[0], reward
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
        vehicle.scooter_inventory_capacity = 0
        vehicle.current_location.scooters = vehicle.current_location.scooters[:1]

        # Get all possible actions
        actions = initial_state.get_possible_actions(
            vehicle,
            number_of_neighbours=5,
        )

        # Test number of actions possible
        self.assertEqual(5, len(actions))

    def test_number_of_actions(self):
        bigger_state = get_initial_state(sample_size=1000, initial_location_depot=False)
        vehicle = bigger_state.vehicles[0]
        vehicle.current_location = random.choice(
            [
                cluster
                for cluster in bigger_state.clusters
                if cluster.number_of_scooters() > 0
            ]
        )
        self.assertLess(
            len(bigger_state.get_possible_actions(vehicle, divide=2)),
            len(bigger_state.get_possible_actions(vehicle)),
        )
        self.assertLess(
            len(bigger_state.get_possible_actions(vehicle, divide=2)),
            len(bigger_state.get_possible_actions(vehicle, divide=4)),
        )


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(40, None, clustering.scripts.get_initial_state(100, 10))

    def test_swap_all_policy(self):
        self.world.policy = decision.SwapAllPolicy()
        vehicle_swap_all_policy = self.world.state.vehicles[0]
        action, _ = self.world.policy.get_best_action(
            self.world, vehicle_swap_all_policy
        )
        self.assertIsInstance(action, Action)
        self.assertEqual(len(action.pick_ups), 0)
        self.assertEqual(len(action.delivery_scooters), 0)


class ValueFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        hyper_params = globals.HyperParameters()
        self.value_function_args = (
            hyper_params.WEIGHT_UPDATE_STEP_SIZE,
            hyper_params.WEIGHT_INITIALIZATION_VALUE,
            hyper_params.DISCOUNT_RATE,
            hyper_params.VEHICLE_INVENTORY_STEP_SIZE,
            hyper_params.LOCATION_REPETITION,
        )
        self.world = World(
            100,
            initial_state=clustering.scripts.get_initial_state(
                1000, 20, initial_location_depot=False
            ),
            policy=None,
        )

    def world_value_function_check(self, value_function):
        # No discount should give reward equal to TD-error
        value_function.setup(self.world.state)
        vehicle = self.world.state.vehicles[0]
        action, _ = decision.policies.SwapAllPolicy().get_best_action(
            self.world, vehicle
        )
        state = copy.deepcopy(self.world.state)
        state_features = value_function.get_state_features(state, vehicle, 0)
        copied_vehicle = copy.deepcopy(vehicle)
        reward, _ = self.world.state.do_action(action, vehicle, self.world.time)
        for i in range(100):
            state_value = value_function.estimate_value(state, copied_vehicle, 0)
            next_state_value = value_function.estimate_value(
                self.world.state, vehicle, self.world.time
            )
            value_function.update_weights(
                current_state_value=state_value,
                current_state_features=state_features,
                next_state_value=next_state_value,
                reward=reward,
            )
        # Check that the fist td errors are bigger than the last
        self.assertLess(
            abs(sum(value_function.td_errors[-3:]) / 3),
            abs(sum(value_function.td_errors[:3]) / 3),
        )

    def test_linear_value_function(self):
        self.world_value_function_check(
            decision.value_functions.LinearValueFunction(*self.value_function_args)
        )

    def test_ann_value_function(self):
        self.world_value_function_check(
            decision.value_functions.ANNValueFunction(
                *self.value_function_args,
                [100, 1000, 100],
            )
        )

    def test_next_state_from_action(self):
        value_function = decision.value_functions.LinearValueFunction(
            *self.value_function_args
        )
        value_function.setup(self.world.state)
        # Record current state
        vehicle = self.world.state.vehicles[0]
        # Scooters is current cluster
        scooters = self.world.state.clusters[
            vehicle.current_location.id
        ].get_swappable_scooters()
        deliver_scooter = random.choice(
            [
                scooter
                for scooter in random.choice(
                    [
                        cluster
                        for cluster in self.world.state.clusters
                        if cluster.id != vehicle.current_location.id
                    ]
                ).scooters
            ]
        )
        vehicle.pick_up(deliver_scooter)
        # Action that does a bit of everything
        action = classes.Action(
            [scooter.id for scooter in scooters[:3]],
            [scooter.id for scooter in scooters[3:10]],
            [deliver_scooter.id],
            random.choice(
                [
                    cluster.id
                    for cluster in self.world.state.clusters
                    if cluster.id != vehicle.current_location.id
                ]
            ),
        )
        function_next_state_features = value_function.convert_next_state_features(
            self.world.state, vehicle, action, self.world.time
        )
        self.world.state.do_action(action, vehicle, self.world.time)
        next_state_features = value_function.convert_state_to_features(
            self.world.state, vehicle, self.world.time
        )
        self.assertEqual(len(function_next_state_features), len(next_state_features))
        for i, value in enumerate(function_next_state_features):
            self.assertAlmostEqual(
                function_next_state_features[i],
                next_state_features[i],
                msg=f"not equal at {i}",
            )


class NeighbourFilteringTests(unittest.TestCase):
    def test_filtering_neighbours(self):
        state = get_initial_state(100, 10)
        vehicle = state.vehicles[0]

        best_neighbours_with_random = filtering_neighbours(
            state,
            vehicle,
            3,
            1,
        )

        # test if the number of neighbours is the same, even though one is random
        self.assertEqual(len(best_neighbours_with_random), 3)

        # Get the three closest neighbors
        three_closest_neighbors = state.get_neighbours(
            vehicle.current_location, is_sorted=True, number_of_neighbours=3
        )

        # Set max deviation in all these clusters
        for cluster in state.clusters:
            if cluster in three_closest_neighbors:
                cluster.ideal_state = 100
                for scooter in cluster.scooters:
                    scooter.battery = 0

        # add one scooter to vehicle inventory so filtering neighbours uses the ideal state deviation filtering method
        vehicle.pick_up(Scooter(0, 0, 0.9, 0))

        best_neighbours = filtering_neighbours(state, vehicle, 3, 0)

        # check if clusters are closest and with the highest deviation -> best neighbours
        for neighbour in best_neighbours:
            self.assertTrue(neighbour in three_closest_neighbors)


if __name__ == "__main__":
    unittest.main()
