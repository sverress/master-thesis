import copy
import random
from classes import World, Cluster, Action
from scenario_simulation.scripts import estimate_reward


class Policy:
    @staticmethod
    def get_best_action(world: World):
        pass


class RandomRolloutPolicy(Policy):
    @staticmethod
    def get_best_action(world: World):
        max_reward = 0
        best_action = None

        # Find all possible actions
        actions = world.state.get_possible_actions(number_of_neighbours=3, divide=2)

        # For every possible action
        for action in actions:
            # Get new state of performing action
            new_state = copy.deepcopy(world.state)
            reward = new_state.do_action(action)

            # Estimate value of making this action, after performing it and calculating the time it takes to perform.
            reward += estimate_reward(new_state, world.get_remaining_time())

            # If the action is better than previous actions, make best_action
            # Add next cluster distance to update shift duration used later.
            if reward >= max_reward:
                max_reward = reward
                best_action = action

        return best_action


class SwapAllPolicy(Policy):
    @staticmethod
    def get_best_action(world: World):
        # Choose a random cluster
        next_cluster: Cluster = random.choice(
            [
                cluster
                for cluster in world.state.clusters
                if cluster.id != world.state.current_cluster.id
            ]
        )

        # Find all scooters that can be swapped there
        swappable_scooters_ids = [
            scooter.id for scooter in next_cluster.get_swappable_scooters()
        ]

        # Calculate how many scooters that can be swapped
        number_of_scooters_to_swap = world.state.get_max_number_of_swaps(next_cluster)

        # Return an action with no re-balancing, only scooter swapping
        return Action(
            battery_swaps=swappable_scooters_ids[:number_of_scooters_to_swap],
            pick_ups=[],
            delivery_scooters=[],
            next_cluster=next_cluster.id,
        )
