import copy
import math
import decision.neighbour_filtering
import classes
from globals import BATTERY_INVENTORY, NUMBER_OF_NEIGHBOURS
import numpy.random as random
import scenario_simulation.scripts


class Policy:
    def get_best_action(self, world, vehicle) -> classes.Action:
        """
        Returns the best action for the input vehicle in the world context
        :param world: world object that contains the whole world state
        :param vehicle: the vehicle to perform an action
        :return: the best action according to the policy
        """
        pass


class TD0Policy(Policy):
    def __init__(self, value_function, epsilon=0.2):
        self.value_function = value_function
        self.epsilon = epsilon

    def get_best_action(self, world, vehicle):
        best_next_state_value = -math.inf
        best_reward = -math.inf
        best_action = None

        # Find all possible actions
        actions = world.state.get_possible_actions(
            vehicle,
            number_of_neighbours=NUMBER_OF_NEIGHBOURS,
            divide=2,
            exclude=world.tabu_list,
            time=world.time,
        )

        if self.epsilon > random.rand():
            return random.choice(actions)
        else:
            for action in actions:
                world_copy = copy.deepcopy(world)
                vehicle_copy = world_copy.state.get_vehicle_by_id(vehicle.id)
                reward = world_copy.state.do_action(action, vehicle_copy)

                next_state_value = self.value_function.estimate_value(
                    world_copy.state, vehicle_copy, world_copy.time
                )

                if next_state_value > best_next_state_value:
                    best_next_state_value = next_state_value
                    best_reward = reward
                    best_action = action

            state_features = self.value_function.create_location_features_combination(
                self.value_function.convert_state_to_features(
                    world.state, vehicle, world.time
                )
            )
            state_value = self.value_function.estimate_value(
                world.state, vehicle, world.time
            )

            self.value_function.update_weights(
                state_features, state_value, best_next_state_value, best_reward
            )

            return best_action

    def __str__(self):
        return "TD=Policy"


class RandomRolloutPolicy(Policy):
    def get_best_action(self, world, vehicle):
        max_reward = -math.inf
        best_action = None

        # Find all possible actions
        actions = world.state.get_possible_actions(
            vehicle,
            number_of_neighbours=NUMBER_OF_NEIGHBOURS,
            divide=2,
            exclude=world.tabu_list,
            time=world.time,
        )

        # For every possible action
        for action in actions:
            # Get new state of performing action
            world_copy = copy.deepcopy(world)
            vehicle_copy = world_copy.state.get_vehicle_by_id(vehicle.id)
            reward = world_copy.state.do_action(action, vehicle_copy)

            # Estimate value of making this action, after performing it and calculating the time it takes to perform.
            reward += world.get_discount() * scenario_simulation.scripts.estimate_reward(
                world_copy, vehicle_copy
            )

            # If the action is better than previous actions, make best_action
            # Add next cluster distance to update shift duration used later.
            if reward >= max_reward:
                max_reward = reward
                best_action = action
        return best_action

    def __str__(self):
        return "RandomRolloutPolicy"


class SwapAllPolicy(Policy):
    def get_best_action(self, world, vehicle):
        # Choose a random cluster
        next_location: classes.Location = decision.neighbour_filtering.filtering_neighbours(
            world.state, vehicle, number_of_neighbours=1, exclude=world.tabu_list
        )[
            0
        ] if vehicle.battery_inventory > BATTERY_INVENTORY * 0.1 else world.state.depots[
            0
        ]

        if vehicle.is_at_depot():
            swappable_scooters_ids = []
            number_of_scooters_to_swap = 0
        else:
            # Find all scooters that can be swapped here
            swappable_scooters_ids = [
                scooter.id
                for scooter in vehicle.current_location.get_swappable_scooters()
            ]

            # Calculate how many scooters that can be swapped
            number_of_scooters_to_swap = vehicle.get_max_number_of_swaps()

        # Return an action with no re-balancing, only scooter swapping
        return classes.Action(
            battery_swaps=swappable_scooters_ids[:number_of_scooters_to_swap],
            pick_ups=[],
            delivery_scooters=[],
            next_location=next_location.id,
        )

    def __str__(self):
        return "SwapAllPolicy"


class RandomActionPolicy(Policy):
    def get_best_action(self, world, vehicle):
        # all possible actions in this state
        possible_actions = world.state.get_possible_actions(
            vehicle, number_of_neighbours=3, exclude=world.tabu_list, time=world.time
        )

        # pick a random action
        return random.choice(possible_actions)

    def __str__(self):
        return "RandomActionPolicy"
