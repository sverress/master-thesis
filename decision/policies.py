import copy
import math
import decision.neighbour_filtering
import classes
import globals
from globals import NUMBER_OF_ROLLOUTS
import numpy.random as random
import scenario_simulation.scripts
import time


class Policy:
    def get_best_action(self, world, vehicle) -> classes.Action:
        """
        Returns the best action for the input vehicle in the world context
        :param world: world object that contains the whole world state
        :param vehicle: the vehicle to perform an action
        :return: the best action according to the policy
        """
        pass

    @staticmethod
    def print_action_stats(
        vehicle: classes.Vehicle, actions_info: [(classes.Action, int, int)],
    ) -> None:
        print(f"\n{vehicle} (#rollouts {NUMBER_OF_ROLLOUTS}):")
        for action, reward, computational_time in actions_info:
            print(
                f"\n{action} Reward - {round(reward,3)} | Comp. time - {round(computational_time, 2)}"
            )
        print("\n----------------------------------------------------------------")


class EpsilonGreedyValueFunctionPolicy(Policy):
    def __init__(self, value_function=None, epsilon=globals.EPSILON):
        self.value_function = value_function
        self.epsilon = epsilon

    def get_best_action(self, world, vehicle):
        # Find all possible actions
        actions = world.state.get_possible_actions(
            vehicle, divide=2, exclude=world.tabu_list, time=world.time,
        )

        # Epsilon greedy choose an action based on value function
        if self.epsilon > random.rand():
            return random.choice(actions)
        else:
            # Create list containing all actions and their rewards and values (action, reward, value_function_value)
            action_info = []
            for action in actions:
                start = time.time()
                world_copy = copy.deepcopy(world)
                vehicle_copy = world_copy.state.get_vehicle_by_id(vehicle.id)
                reward = world_copy.state.do_action(action, vehicle_copy)
                next_state_value = self.value_function.estimate_value(
                    world_copy.state, vehicle_copy, world_copy.time
                )
                stop = time.time()
                action_info.append((action, reward, next_state_value, stop - start))
            # Find the action with the highest reward and future expected reward - reward + value function next state
            (
                best_action,
                best_action_reward,
                best_action_next_state_value,
                best_action_time,
            ) = max(action_info, key=lambda pair: pair[1] + pair[2])
            if world.verbose:
                Policy.print_action_stats(
                    vehicle,
                    [
                        (action, reward + next_state_value, elapsed_time)
                        for action, reward, next_state_value, elapsed_time in action_info
                    ],
                )
            state_features = self.value_function.get_state_features(
                world.state, vehicle, world.time
            )
            state_value = self.value_function.estimate_value(
                world.state, vehicle, world.time, state_features=state_features
            )

            self.value_function.update_weights(
                state_features,
                state_value,
                best_action_next_state_value,
                best_action_reward,
            )

            return best_action


class RandomRolloutPolicy(Policy):
    def get_best_action(self, world, vehicle):
        max_reward = -math.inf
        best_action = None

        # Find all possible actions
        actions = world.state.get_possible_actions(
            vehicle, divide=2, exclude=world.tabu_list, time=world.time,
        )
        actions_info = []
        # For every possible action
        for action in actions:
            start = time.time()
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

            actions_info.append((action, reward, time.time() - start))

        if world.verbose:
            Policy.print_action_stats(vehicle, actions_info)

        return best_action


class SwapAllPolicy(Policy):
    def get_best_action(self, world, vehicle):
        # Choose a random cluster
        next_location: classes.Location = decision.neighbour_filtering.filtering_neighbours(
            world.state,
            vehicle,
            number_of_neighbours=1,
            exclude=world.tabu_list,
            max_swaps=vehicle.get_max_number_of_swaps(),
        )[
            0
        ] if vehicle.battery_inventory > vehicle.battery_inventory_capacity * 0.1 else world.state.depots[
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


class RandomActionPolicy(Policy):
    def get_best_action(self, world, vehicle):
        # all possible actions in this state
        possible_actions = world.state.get_possible_actions(
            vehicle, exclude=world.tabu_list, time=world.time
        )

        # pick a random action
        return random.choice(possible_actions)
