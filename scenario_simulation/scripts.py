import copy
import decision
from globals import ITERATION_LENGTH_MINUTES, LOST_TRIP_REWARD, NUMBER_OF_ROLLOUTS


def estimate_reward(
    world, vehicle, number_of_simulations=NUMBER_OF_ROLLOUTS,
):
    """
    Does n times scenario simulations and returns the highest conducted reward from simulation
    :param vehicle: vehicle to estimate reward for
    :param world: snapshot copy of world
    :param number_of_simulations: int - number of simulations to be performed (default = 10)
    :return: int - maximum reward from simulations
    """

    all_rewards = []
    random_action_policy = decision.RandomActionPolicy()
    # Do n scenario simulations
    for i in range(number_of_simulations):
        simulation_counter = 1
        next_is_vehicle_action = True
        world_copy = copy.deepcopy(world)
        vehicle_copy = [
            new_vehicle_copy
            for new_vehicle_copy in world_copy.state.vehicles
            if vehicle.id == new_vehicle_copy.id
        ][0]
        # Simulate until shift ends
        while world_copy.time < world_copy.shift_duration:
            if next_is_vehicle_action:
                action = random_action_policy.get_best_action(world_copy, vehicle_copy)
                previous_cluster_id = vehicle_copy.current_location.id
                world_copy.add_reward(
                    world_copy.get_discount()
                    * world_copy.state.do_action(action, vehicle_copy)
                )
                world_copy.time = world_copy.time + action.get_action_time(
                    world_copy.state.get_distance_id(
                        previous_cluster_id, action.next_location
                    )
                )

            else:
                _, _, lost_demand = world_copy.state.system_simulate()
                world_copy.add_reward(lost_demand * LOST_TRIP_REWARD)
                simulation_counter += 1
            next_is_vehicle_action = (
                world_copy.time < simulation_counter * ITERATION_LENGTH_MINUTES
            )
        all_rewards.append(world_copy.get_total_reward())

    return sum(all_rewards) / len(all_rewards)
