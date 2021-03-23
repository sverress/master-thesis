import copy
import classes
import decision.policies as policies
from globals import ITERATION_LENGTH_MINUTES, LOST_TRIP_REWARD, NUMBER_OF_ROLLOUTS


def estimate_reward(
    state, remaining_shift_duration: int, number_of_simulations=NUMBER_OF_ROLLOUTS,
):
    """
    Does n times scenario simulations and returns the highest conducted reward from simulation
    :param state: State - state to de the simulations from
    :param remaining_shift_duration: int - time left on shift = length of simulation
    :param number_of_simulations: int - number of simulations to be performed (default = 10)
    :return: int - maximum reward from simulations
    """

    all_rewards = []

    # Do n scenario simulations
    for i in range(number_of_simulations):
        simulation_counter = 1
        world = classes.World(
            remaining_shift_duration, initial_state=copy.deepcopy(state)
        )
        next_is_vehicle_action = True
        # Simulate until shift ends
        while world.time < remaining_shift_duration:
            if next_is_vehicle_action:
                action = policies.RandomActionPolicy.get_best_action(world)
                world.add_reward(world.get_discount() * world.state.do_action(action))
                world.time = world.time + action.get_action_time(
                    world.state.get_distance_id(
                        world.state.current_cluster.id, action.next_cluster
                    )
                )
            else:
                _, _, lost_demand = world.state.system_simulate()
                world.add_reward(lost_demand * LOST_TRIP_REWARD)
                simulation_counter += 1
            next_is_vehicle_action = (
                world.time < simulation_counter * ITERATION_LENGTH_MINUTES
            )

        all_rewards.append(world.get_total_reward())

    return max(all_rewards)
