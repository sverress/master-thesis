import copy

import classes
import clustering.scripts
import decision.value_functions
import training_simulation.scripts
from progress.bar import IncrementalBar

import globals


def train_value_function(
    world, save_suffix="", scenario_training=True, epsilon_decay=True
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
    world.policy.epsilon = world.INITIAL_EPSILON if epsilon_decay else world.EPSILON
    for shift in range(number_of_shifts + 1):
        policy_world = copy.deepcopy(world)
        policy_world.policy.value_function.update_shifts_trained(shift)
        if epsilon_decay:
            policy_world.policy.epsilon -= (
                world.INITIAL_EPSILON - world.FINAL_EPSILON
            ) / number_of_shifts
        if shift % world.TRAINING_SHIFTS_BEFORE_SAVE == 0:
            policy_world.save_world(
                cache_directory=world.get_train_directory(save_suffix), suffix=shift
            )

        # avoid running the world after the last model is saved
        if shift != number_of_shifts:
            if scenario_training:
                training_simulation.scripts.training_simulation(policy_world)
            else:
                policy_world.run()

            if hasattr(policy_world.policy, "value_function"):
                policy_world.policy.value_function.reset_eligibilities()
            world.policy = policy_world.policy
            progress_bar.next()


if __name__ == "__main__":
    SAMPLE_SIZE = 2500
    NUMBER_OF_CLUSTERS = 30
    standard_parameters = globals.HyperParameters()
    world_to_analyse = classes.World(
        60,
        None,
        clustering.scripts.get_initial_state(
            SAMPLE_SIZE,
            NUMBER_OF_CLUSTERS,
            number_of_vans=4,
            number_of_bikes=0,
        ),
        verbose=False,
        visualize=False,
        MODELS_TO_BE_SAVED=3,
        TRAINING_SHIFTS_BEFORE_SAVE=1,
    )
    world_to_analyse.policy = world_to_analyse.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )

    train_value_function(world_to_analyse)
