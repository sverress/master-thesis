import os
import unittest
import random
import classes
import clustering.scripts
import decision
import decision.value_functions
import globals


class WorldTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.world = classes.World(
            40,
            policy=decision.SwapAllPolicy(),
            initial_state=clustering.scripts.get_initial_state(100, 10),
            visualize=False,
        )

    def test_run(self):
        self.world.stack = [classes.Event(time) for time in range(10, 41, 10)]
        self.world.run()

    def test_add_event(self):
        # Clear initial stack
        self.world.stack = []
        shuffled_events = [classes.Event(time) for time in range(10, 41, 10)]
        random.shuffle(shuffled_events)
        for event in shuffled_events:
            self.world.add_event(event)
        self.assertSequenceEqual(
            [event.time for event in self.world.stack], range(10, 41, 10)
        )

    def test_run_with_initial_stack(self):
        self.world.run()

    def test_tabu_list(self):
        # Clear initial stack
        self.world.stack = []
        # Perform Vehicle arrival event
        arrival_event = classes.VehicleArrival(0, 0, False)
        arrival_event.perform(self.world)
        vehicle = [vehicle for vehicle in self.world.state.vehicles if vehicle.id == 0][
            0
        ]
        # Record the next location
        first_vehicle_location = vehicle.current_location.id

        # Check that next vehicle location is in tabu list
        self.assertIn(first_vehicle_location, self.world.tabu_list)
        # Check that exclude works in next vehicle location not in possible actions
        self.assertNotIn(
            first_vehicle_location,
            [
                action.next_location
                for action in self.world.state.get_possible_actions(
                    vehicle, exclude=self.world.tabu_list
                )
            ],
        )
        # Perform vehicle next vehicle arrival event
        self.world.stack.pop().perform(self.world)
        # Check that the old vehicle location is not in the tabu list
        self.assertNotIn(first_vehicle_location, self.world.tabu_list)

    def test_save_world(self):
        # Change weights in value function
        self.world.policy = self.world.set_policy(
            decision.EpsilonGreedyValueFunctionPolicy(
                decision.value_functions.LinearValueFunction()
            )
        )
        self.world.policy.value_function.weights[0] = 0.1
        # Save, load and delete world object
        filepath = f"{globals.WORLD_CACHE_DIR}/{self.world.get_filename()}"
        self.world.save_world()
        file_world = classes.World.load(filepath)
        self.assertEqual(0.1, file_world.policy.value_function.weights[0])
        os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
