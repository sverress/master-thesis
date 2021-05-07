import copy

import classes
import clustering.scripts
import decision.value_functions
from progress.bar import IncrementalBar

import globals


def train_value_function(
    world, save_suffix="",
):
    progress_bar = IncrementalBar(
        "Training value function",
        check_tty=False,
        max=(world.TRAINING_SHIFTS_BEFORE_SAVE * world.MODELS_TO_BE_SAVED),
        suffix="%(percent)d%% - ETA %(eta)ds",
    )
    print(
        f"-------------------- {world.policy.value_function.__str__()} training --------------------"
    )
    number_of_shifts = world.TRAINING_SHIFTS_BEFORE_SAVE * world.MODELS_TO_BE_SAVED
    for shift in range(number_of_shifts + 1):
        policy_world = copy.deepcopy(world)
        policy_world.policy.value_function.update_shifts_trained(shift)

        if shift % world.TRAINING_SHIFTS_BEFORE_SAVE == 0:
            policy_world.save_world(
                cache_directory=world.get_train_directory(save_suffix), suffix=shift
            )

        if shift != number_of_shifts:
            # avoid running the world after the last model is saved
            policy_world.run()
            world.policy = policy_world.policy
            progress_bar.next()


if __name__ == "__main__":
    SAMPLE_SIZE = 2500
    NUMBER_OF_CLUSTERS = 30
    standard_parameters = globals.HyperParameters()
    world_to_analyse = classes.World(
        5,
        None,
        clustering.scripts.get_initial_state(
            SAMPLE_SIZE, NUMBER_OF_CLUSTERS, number_of_vans=1, number_of_bikes=0,
        ),
        verbose=False,
        visualize=False,
        NUMBER_OF_NEIGHBOURS=4,
        MODELS_TO_BE_SAVED=3,
        TRAINING_SHIFTS_BEFORE_SAVE=1,
    )
    world_to_analyse.policy = world_to_analyse.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.LinearValueFunction,
    )
    train_value_function(world_to_analyse)
