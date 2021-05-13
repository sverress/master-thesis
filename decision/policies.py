import decision.neighbour_filtering
import classes
import numpy.random as random
import abc


class Policy(abc.ABC):
    def __init__(
        self,
        get_possible_actions_divide,
        number_of_neighbors,
    ):
        self.get_possible_actions_divide = get_possible_actions_divide
        self.number_of_neighbors = number_of_neighbors

    @abc.abstractmethod
    def get_best_action(self, world, vehicle):
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
        world,
        vehicle: classes.Vehicle,
        actions_info: [(classes.Action, int, int)],
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
        self,
        get_possible_actions_divide,
        number_of_neighbors,
        epsilon,
        value_function,
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
        state = world.state
        # Epsilon greedy choose an action based on value function
        if self.epsilon > random.rand():
            return random.choice(actions)
        else:
            # Create list containing all actions and their rewards and values (action, reward, value_function_value)
            action_info = []
            # Cache current states in state
            cache = [cluster.get_current_state() for cluster in state.clusters], [
                cluster.get_available_scooters() for cluster in state.clusters
            ]
            # Generate the state features of the current state
            state_features = self.value_function.get_state_features(
                world.state, vehicle, world.time, cache
            )
            state_value = self.value_function.estimate_value_from_state_features(
                state_features
            )
            expected_lost_trip_reward = state.get_expected_lost_trip_reward(
                world.LOST_TRIP_REWARD, exclude=vehicle.current_location.id
            )
            for action in actions:
                # Get the distance from current cluster to the new destination cluster
                action_distance = state.get_distance(
                    vehicle.current_location.id, action.next_location
                )
                # Generate the features for this new state after the action
                next_state_features = self.value_function.get_next_state_features(
                    state,
                    vehicle,
                    action,
                    world.time + action.get_action_time(action_distance),
                    cache,
                )
                # Calculate the expected future reward of being in this new state
                next_state_value = (
                    self.value_function.estimate_value_from_state_features(
                        next_state_features
                    )
                )

                action_info.append(
                    (
                        action,
                        action.get_reward(vehicle, world.LOST_TRIP_REWARD),
                        next_state_value,
                    )
                )

            # Find the action with the highest reward and future expected reward - reward + value function next state
            best_action, reward, next_state_value = max(
                action_info, key=lambda pair: pair[1] + pair[2]
            )

            self.value_function.update_weights(
                state_features,
                state_value,
                reward + expected_lost_trip_reward,
                next_state_value,
            )

            return best_action

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

    def get_best_action(self, world, vehicle):
        return classes.Action([], [], [], 0)


class RebalancingPolicy(Policy):
    def __init__(self):
        super().__init__(0, 0)

    def get_best_action(self, world, vehicle):
        vehicle_has_scooter_inventory = len(vehicle.scooter_inventory) > 0
        if vehicle.is_at_depot():
            scooters_to_deliver = []
            scooters_to_pickup = []
            number_of_scooters_to_pick_up = 0
            number_of_scooters_to_swap = 0
            scooters_to_swap = []
        else:
            # If vehicle has scooter inventory, deliver all scooters and swap all swappable scooters
            if vehicle_has_scooter_inventory:
                # Deliver all scooters in scooter inventory, and don't pick up any new scooters
                scooters_to_deliver = [
                    scooter.id for scooter in vehicle.scooter_inventory
                ]
                scooters_to_pickup = []
                number_of_scooters_to_pick_up = 0
                # Swap as many scooters as possible as this cluster most likely needs it
                swappable_scooters = vehicle.current_location.get_swappable_scooters()
                number_of_scooters_to_swap = min(
                    vehicle.battery_inventory, len(swappable_scooters)
                )
                scooters_to_swap = [scooter.id for scooter in swappable_scooters][
                    :number_of_scooters_to_swap
                ]
            else:
                # Pick up as many scooters as possible, the min(scooter capacity, deviation from ideal state)
                number_of_scooters_to_pick_up = min(
                    vehicle.scooter_inventory_capacity,
                    len(vehicle.current_location.scooters)
                    - vehicle.current_location.ideal_state,
                )
                scooters_to_pickup = [
                    scooter.id for scooter in vehicle.current_location.scooters
                ][:number_of_scooters_to_pick_up]
                # Do not swap any scooters in a cluster with a lot of scooters
                scooters_to_swap = []
                number_of_scooters_to_swap = 0
                # There are no scooters to deliver due to empty inventory
                scooters_to_deliver = []

        def get_next_location_id(is_finding_positive_deviation):
            return sorted(
                [
                    cluster
                    for cluster in world.state.clusters
                    if cluster.id != vehicle.current_location.id
                    and cluster.id not in world.tabu_list
                ],
                key=lambda cluster: len(cluster.get_available_scooters())
                - cluster.ideal_state,
                reverse=is_finding_positive_deviation,
            )[0].id

        # If scooters has under 10% battery inventory, go to depot.
        if (
            vehicle.battery_inventory
            - number_of_scooters_to_swap
            - number_of_scooters_to_pick_up
            < vehicle.battery_inventory_capacity * 0.1
        ) and not vehicle.is_at_depot():
            next_location_id = world.state.depots[0].id
        else:
            """
            If vehicle has scooter inventory upon arrival,
            go to new positive deviation cluster to pick up new scooters.
            If there are no scooter inventory, go to cluster where you
            can drop off scooters picked up in this cluster, ergo negative deviation cluster.
            If, however, you are in the depot, you should do the opposite as the depot does not
            change the scooter inventory.
            """
            visit_positive_deviation_cluster_next = (
                vehicle_has_scooter_inventory
                if not vehicle.is_at_depot()
                else not vehicle_has_scooter_inventory
            )
            next_location_id = get_next_location_id(
                visit_positive_deviation_cluster_next
            )

        return classes.Action(
            scooters_to_swap,
            scooters_to_pickup,
            scooters_to_deliver,
            next_location_id,
        )
