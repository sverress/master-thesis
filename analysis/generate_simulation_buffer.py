import copy
import os
import pickle

from progress.bar import IncrementalBar

import classes
import decision.value_functions
import training_simulation.scripts

import clustering.scripts


def generate_simulation_buffer(world, number_of_shifts):
    """
    Generates a .npy file with all cases from using epsilon greedy on the rebalancing policy
    """
    progress_bar = IncrementalBar(
        "Generate cases",
        check_tty=False,
        max=number_of_shifts - 1,
        suffix="%(percent)d%% - ETA %(eta)ds",
    )
    world.policy.epsilon = world.INITIAL_EPSILON
    for i in range(number_of_shifts):
        world_copy = copy.deepcopy(world)
        world_copy.policy.epsilon -= (
            world.INITIAL_EPSILON - world.FINAL_EPSILON
        ) / number_of_shifts
        training_simulation.scripts.training_simulation(world_copy)
        progress_bar.next()

    # If there is no world_cache directory, create it
    if not os.path.exists("transitions"):
        os.makedirs("transitions")

    with open(f"transitions/{world.get_filename()}_transitions.pickle", "wb") as file:
        pickle.dump(world.policy.value_function.replay_buffer, file)


if __name__ == "__main__":
    world = classes.World(
        960,
        None,
        clustering.scripts.get_initial_state(2500, 50, number_of_vans=2),
        verbose=False,
        visualize=False,
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyRebalancingPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )
    generate_simulation_buffer(
        world,
        20_000,
    )
