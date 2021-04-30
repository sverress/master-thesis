import copy
import globals
import classes
import clustering.scripts
import decision.value_functions
from progress.bar import IncrementalBar


def train_value_function(
    world,
    training_shifts_before_save=globals.TRAINING_SHIFTS_BEFORE_SAVE,
    models_to_be_saved=globals.MODELS_TO_BE_SAVED,
):
    progress_bar = IncrementalBar(
        "Training value function",
        check_tty=False,
        max=(training_shifts_before_save * models_to_be_saved),
        suffix="%(percent)d%% - ETA %(eta)ds",
    )
    print(
        f"-------------------- {world.policy.value_function.__str__()} training --------------------"
    )
    number_of_shifts = training_shifts_before_save * models_to_be_saved
    for shift in range(number_of_shifts + 1):
        policy_world = copy.deepcopy(world)
        policy_world.policy.value_function.update_shifts_trained(shift)

        if shift % training_shifts_before_save == 0:
            policy_world.save_world([world.get_train_directory(), shift])

        if shift != number_of_shifts:
            # avoid running the world after the last model is saved
            policy_world.run()
            world.policy = policy_world.policy
            progress_bar.next()


if __name__ == "__main__":
    SAMPLE_SIZE = 100
    NUMBER_OF_CLUSTERS = 10

    POLICY = decision.EpsilonGreedyValueFunctionPolicy(
        decision.value_functions.LinearValueFunction()
    )

    world_to_analyse = classes.World(
        6,
        None,
        clustering.scripts.get_initial_state(
            SAMPLE_SIZE, NUMBER_OF_CLUSTERS, number_of_vans=2, number_of_bikes=0
        ),
        verbose=False,
        visualize=False,
    )
    world_to_analyse.policy = world_to_analyse.set_policy(POLICY)
    train_value_function(world_to_analyse)
