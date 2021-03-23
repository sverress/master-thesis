import copy
import decision.neighbour_filtering
import classes
import numpy.random as random
import scenario_simulation.scripts


class Policy:
    @staticmethod
    def get_best_action(world):
        pass


class RandomRolloutPolicy(Policy):
    @staticmethod
    def get_best_action(world):
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
        next_cluster: classes.Cluster = decision.neighbour_filtering.filtering_neighbours(
            world.state, number_of_neighbours=1
        )[
            0
        ]

        # Find all scooters that can be swapped here
        swappable_scooters_ids = [
            scooter.id
            for scooter in world.state.current_cluster.get_swappable_scooters()
        ]

        # Calculate how many scooters that can be swapped
        number_of_scooters_to_swap = world.state.get_max_number_of_swaps(
            world.state.current_cluster
        )

        # Return an action with no re-balancing, only scooter swapping
        return classes.Action(
            battery_swaps=swappable_scooters_ids[:number_of_scooters_to_swap],
            pick_ups=[],
            delivery_scooters=[],
            next_cluster=next_cluster.id,
        )


class RandomActionPolicy(Policy):
    @staticmethod
    def get_best_action(world):
        # all possible actions in this state
        possible_actions = world.state.get_possible_actions(number_of_neighbours=3)

        # pick a random action
        return random.choice(possible_actions)
