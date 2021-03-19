import copy
import classes
from globals import ITERATION_LENGTH_MINUTES, NUMBER_OF_ROLLOUTS
import decision.policies as policies


def estimate_reward(
    state: classes.State,
    remaining_shift_duration: int,
    number_of_simulations=NUMBER_OF_ROLLOUTS,
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
        iteration_counter = 0
        world = classes.World(
            remaining_shift_duration, initial_state=copy.deepcopy(state)
        )
        # Simulate until shift ends
        while iteration_counter * ITERATION_LENGTH_MINUTES < remaining_shift_duration:
            world.add_reward(
                world.state.do_action(
                    policies.RandomActionPolicy.get_best_action(world)
                )
            )
            world.state.system_simulate()
            iteration_counter += 1

        all_rewards.append(world.get_total_reward())

    return max(all_rewards)
