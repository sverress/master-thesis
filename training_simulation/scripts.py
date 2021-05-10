from globals import ITERATION_LENGTH_MINUTES


def training_simulation(world):
    """
    Does scenario simulations until shift end
    :param world: snapshot copy of world
    :return: world object after shift
    """
    simulation_counter = 1
    next_is_vehicle_action = True
    # list of vehicle times for next arrival
    vehicle_times = [0] * len(world.state.vehicles)
    action_infos = []
    while world.time < world.shift_duration:
        if next_is_vehicle_action:
            # choosing the vehicle with the earliest arrival time (index-method is choosing the first if multiple equal)
            vehicle_index = vehicle_times.index(min(vehicle_times))
            # fetching the vehicle
            current_vehicle = world.state.vehicles[vehicle_index]

            # Remove current vehicle state from tabu list
            world.tabu_list = [
                cluster_id
                for cluster_id in world.tabu_list
                if cluster_id != current_vehicle.current_location.id
            ]

            # getting the best action
            action, action_info = world.policy.get_best_action(world, current_vehicle)

            # random action -> action_info = None. Not include random action in training
            if action_info:
                action_infos.append(action_info)

            action_time = action.get_action_time(
                world.state.get_distance(
                    current_vehicle.current_location.id, action.next_location
                )
            )
            # Performing the best action
            _, refill_time = world.state.do_action(action, current_vehicle, world.time)

            action_time += refill_time

            # Add next vehicle location to tabu list if its not a depot
            if not current_vehicle.is_at_depot():
                world.tabu_list.append(action.next_location)

            # updating the current vehicle time to the next arrival
            vehicle_times[vehicle_index] += action_time
            # setting the world time to the next vehicle arrival
            world.time = world.time + min(vehicle_times)

        else:
            # performing a scooter trips simulation
            _, _, lost_demand = world.system_simulate()

            # if some actions has been made since last simulation, update value function
            if len(action_infos) > 0:
                action_infos = [
                    (
                        state_value,
                        reward
                        + (
                            lost_demand * world.LOST_TRIP_REWARD
                        ),  # Add lost trip reward from system simulation
                        next_state_value,
                        state_features,
                    )
                    for state_value, next_state_value, reward, state_features in action_infos
                ]
                # Update value function
                world.policy.value_function.batch_update_weights(action_infos)
                # Clear case base TODO: Add action info to a case base, and let case stay in the case base for longer
                action_infos = []

            simulation_counter += 1
        # deciding if the next thing to do is a vehicle arrival or a system simulation
        next_is_vehicle_action = (
            world.time < simulation_counter * ITERATION_LENGTH_MINUTES
        )

    return world


if __name__ == "__main__":

    import classes
    import decision.value_functions
    import clustering.scripts

    world_to_analyse = classes.World(
        240,
        None,
        clustering.scripts.get_initial_state(
            2500, 30, number_of_vans=3, number_of_bikes=0,
        ),
        verbose=False,
        visualize=False,
    )
    world_to_analyse.policy = world_to_analyse.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.LinearValueFunction,
    )

    training_simulation(world_to_analyse)
