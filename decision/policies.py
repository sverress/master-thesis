import copy
import decision.neighbour_filtering
import classes
import numpy.random as random
import abc


class Policy(abc.ABC):
    def __init__(
        self, get_possible_actions_divide, number_of_neighbors,
    ):
        self.get_possible_actions_divide = get_possible_actions_divide
        self.number_of_neighbors = number_of_neighbors

    @abc.abstractmethod
    def get_best_action(self, world, vehicle) -> classes.Action:
        """
        Returns the best action for the input vehicle in the world context
        :param world: world object that contains the whole world state
        :param vehicle: the vehicle to perform an action
        :return: the best action according to the policy
        """
        pass

    def setup_from_state(self, state):
        """
        Function to be called after association with a state object is created.
        Nice place to setup value functions.
        :param state: state object associated with policy
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @staticmethod
    def print_action_stats(
        world, vehicle: classes.Vehicle, actions_info: [(classes.Action, int, int)],
    ) -> None:
        if world.verbose:
            print(f"\n{vehicle}:")
            for action, reward, computational_time in actions_info:
                print(
                    f"\n{action} Reward - {round(reward, 3)} | Comp. time - {round(computational_time, 2)}"
                )
            print("\n----------------------------------------------------------------")


class EpsilonGreedyValueFunctionPolicy(Policy):
    """
    Chooses an action based on a epsilon greedy policy. Will update weights after chosen action
    """

    def __init__(
        self, get_possible_actions_divide, number_of_neighbors, epsilon, value_function,
    ):
        super().__init__(get_possible_actions_divide, number_of_neighbors)
        self.value_function = value_function
        self.epsilon = epsilon

    def get_best_action(self, world, vehicle):
        # Find all possible actions
        actions = world.state.get_possible_actions(
            vehicle,
            divide=self.get_possible_actions_divide,
            exclude=world.tabu_list,
            time=world.time,
            number_of_neighbours=self.number_of_neighbors,
        )

        # Epsilon greedy choose an action based on value function
        if self.epsilon > random.rand():
            return random.choice(actions)
        else:
            # Create list containing all actions and their rewards and values (action, reward, value_function_value)
            action_info = []
            # Generate the state features of the current state
            state_features = self.value_function.get_state_features(
                world.state, vehicle, world.time
            )
            state_value = self.value_function.estimate_value_from_state_features(
                state_features
            )
            for action in actions:
                # Copy state to avoid pointer issue
                state_copy = copy.deepcopy(world.state)
                # Get the relevant vehicle from the state copy
                vehicle_copy = state_copy.get_vehicle_by_id(vehicle.id)
                # Perform the action and record the reward
                reward, _ = state_copy.do_action(action, vehicle_copy, world.time)
                # Get the distance from current cluster to the new destination cluster
                action_distance = state_copy.get_distance(
                    vehicle.current_location.id, action.next_location
                )
                # Generate the features for this new state after the action
                next_state_features = self.value_function.get_state_features(
                    state_copy,
                    vehicle_copy,
                    world.time + action.get_action_time(action_distance),
                )
                # Calculate the expected future reward of being in this new state
                next_state_value = self.value_function.estimate_value_from_state_features(
                    next_state_features
                )

                action_info.append((action, next_state_value, reward))

            # Find the action with the highest reward and future expected reward - reward + value function next state
            best_action, *rest = max(action_info, key=lambda pair: pair[1] + pair[2])

            # Best action, (reward, next state estimated value, state features)
            return best_action, (state_value, *rest, state_features)

    def setup_from_state(self, state):
        self.value_function.setup(state)

    def __str__(self):
        return f"EpsilonGreedyPolicy w/ {self.value_function.__str__()}"


class SwapAllPolicy(Policy):
    def __init__(self):

        super().__init__(1, 1)

    def get_best_action(self, world, vehicle):
        # Choose a random cluster
        next_location: classes.Location = (
            decision.neighbour_filtering.filtering_neighbours(
                world.state,
                vehicle,
                self.number_of_neighbors,
                0,
                exclude=world.tabu_list,
                max_swaps=vehicle.get_max_number_of_swaps(),
            )[0]
            if vehicle.battery_inventory > vehicle.battery_inventory_capacity * 0.1
            else world.state.depots[0]
        )

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
    def __init__(self, get_possible_actions_divide, number_of_neighbors):
        super().__init__(get_possible_actions_divide, number_of_neighbors)

    def get_best_action(self, world, vehicle):
        # all possible actions in this state
        possible_actions = world.state.get_possible_actions(
            vehicle,
            exclude=world.tabu_list,
            time=world.time,
            divide=self.get_possible_actions_divide,
            number_of_neighbours=self.number_of_neighbors,
        )

        # pick a random action
        return random.choice(possible_actions)


class DoNothing(Policy):
    def __init__(self):
        super().__init__(0, 0)

    def get_best_action(self, world, vehicle) -> classes.Action:
        return classes.Action([], [], [], 0)
