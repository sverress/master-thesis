from globals import ITERATION_LENGTH_MINUTES


def training_simulation(world):
    """
    Does scenario simulations until shift end
    :param world: snapshot copy of world
    :return: world object after shift
    """
    simulation_counter = 1
    lost_demand = 0
    # list to hold previous sap-features
    vehicle_sap_features = [None] * len(world.state.vehicles)
    next_is_vehicle_action = True
    # list of vehicle times for next arrival
    vehicle_times = [0] * len(world.state.vehicles)
    # list to hold previous action rewards
    vehicle_rewards = [0] * len(world.state.vehicles)

    while world.time < world.shift_duration:
        if next_is_vehicle_action:
            # choosing the vehicle with the earliest arrival time (index-method is choosing the first if multiple equal)
            vehicle_index = vehicle_times.index(min(vehicle_times))
            # fetching the vehicle
            current_vehicle = world.state.vehicles[vehicle_index]

            # getting the best action and setting this to current vehicle action
            action, sap_features = world.policy.get_best_action(world, current_vehicle)

            # Remove current vehicle state from tabu list
            world.tabu_list = [
                cluster_id
                for cluster_id in world.tabu_list
                if cluster_id != current_vehicle.current_location.id
            ]

            action_time = action.get_action_time(
                world.state.get_distance(
                    current_vehicle.current_location.id, action.next_location
                )
            )

            reward = action.get_reward(
                current_vehicle,
                world.DEPOT_REWARD,
                world.VEHICLE_INVENTORY_STEP_SIZE,
                world.PICK_UP_REWARD,
            )

            # Performing the best action and adding refill_time to action_time
            action_time += world.state.do_action(action, current_vehicle, world.time)

            # Add next vehicle location to tabu list if its not a depot
            if not current_vehicle.is_at_depot():
                world.tabu_list.append(action.next_location)

            # updating the current vehicle time to the next arrival
            vehicle_times[vehicle_index] += action_time
            # setting the world time to the next vehicle arrival
            world.time = world.time + min(vehicle_times)

            if sap_features[vehicle_index]:
                world.policy.value_function.replay_buffer.append(
                    (
                        vehicle_sap_features[vehicle_index],
                        vehicle_rewards[vehicle_index]
                        - lost_demand * world.LOST_TRIP_REWARD,
                        sap_features,
                    )
                )

            vehicle_sap_features[vehicle_index] = sap_features
            vehicle_rewards[vehicle_index] = reward
        else:
            # performing a scooter trips simulation
            _, _, lost_demands = world.system_simulate()
            lost_demand = (
                sum(map(lambda lost_trips: lost_trips[0], lost_demands))
                if len(lost_demands) > 0
                else 0
            )
            simulation_counter += 1
        # deciding if the next thing to do is a vehicle arrival or a system simulation
        next_is_vehicle_action = (
            world.time < simulation_counter * ITERATION_LENGTH_MINUTES
        )

    # adding the last sap to replay buffer and setting next-sap too None -> next-sap-value = 0 in train method
    for vehicle_index in range(len(world.state.vehicles)):
        world.policy.value_function.replay_buffer.append(
            (
                vehicle_sap_features[vehicle_index],
                vehicle_rewards[vehicle_index] - lost_demand * world.LOST_TRIP_REWARD,
                None,
            )
        )

    return world


if __name__ == "__main__":

    import classes
    import decision.value_functions
    import clustering.scripts

    world_to_analyse = classes.World(
        960,
        None,
        clustering.scripts.get_initial_state(
            2500,
            30,
            number_of_vans=3,
            number_of_bikes=0,
        ),
        verbose=False,
        visualize=False,
    )
    world_to_analyse.policy = world_to_analyse.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    training_simulation(world_to_analyse)
    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
