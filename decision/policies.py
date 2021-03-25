import copy
import math
import decision.neighbour_filtering
import classes
from globals import BATTERY_INVENTORY, NUMBER_OF_NEIGHBOURS
import numpy.random as random
import scenario_simulation.scripts


class Policy:
    @staticmethod
    def get_best_action(world):
        pass


class RandomRolloutPolicy(Policy):
    @staticmethod
    def get_best_action(world):
        max_reward = -math.inf
        best_action = None

        # Find all possible actions
        actions = world.state.get_possible_actions(
            number_of_neighbours=NUMBER_OF_NEIGHBOURS, divide=2, time=world.time
        )

        # For every possible action
        for action in actions:
            # Get new state of performing action
            new_state = copy.deepcopy(world.state)
            reward = new_state.do_action(action)

            # Estimate value of making this action, after performing it and calculating the time it takes to perform.
            reward += world.get_discount() * scenario_simulation.scripts.estimate_reward(
                new_state, world.get_remaining_time()
            )

            # If the action is better than previous actions, make best_action
            # Add next cluster distance to update shift duration used later.
            if reward >= max_reward:
                max_reward = reward
                best_action = action

        return best_action


class SwapAllPolicy(Policy):
    @staticmethod
    def get_best_action(world):
        # Choose a random cluster
        next_location: classes.Location = decision.neighbour_filtering.filtering_neighbours(
            world.state, number_of_neighbours=1,
        )[
            0
        ] if world.state.vehicle.battery_inventory > BATTERY_INVENTORY * 0.1 else world.state.depots[
            0
        ]

        if world.state.is_at_depot():
            swappable_scooters_ids = []
            number_of_scooters_to_swap = 0
        else:
            # Find all scooters that can be swapped here
            swappable_scooters_ids = [
                scooter.id
                for scooter in world.state.current_location.get_swappable_scooters()
            ]

            # Calculate how many scooters that can be swapped
            number_of_scooters_to_swap = world.state.get_max_number_of_swaps(
                world.state.current_location
            )

        # Return an action with no re-balancing, only scooter swapping
        return classes.Action(
            battery_swaps=swappable_scooters_ids[:number_of_scooters_to_swap],
            pick_ups=[],
            delivery_scooters=[],
            next_location=next_location.id,
        )


class RandomActionPolicy(Policy):
    @staticmethod
    def get_best_action(world):
        # all possible actions in this state
        possible_actions = world.state.get_possible_actions(
            number_of_neighbours=3, time=world.time
        )

        # pick a random action
        return random.choice(possible_actions)
