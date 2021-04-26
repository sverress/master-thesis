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
        "Running World",
        check_tty=False,
        max=(training_shifts_before_save * models_to_be_saved),
        suffix="%(percent)d%% - ETA %(eta)ds",
    )
    print(
        f"-------------------- {world.policy.roll_out_policy.value_function} training --------------------"
    )
    for shift in range(training_shifts_before_save * models_to_be_saved):
        policy_world = copy.deepcopy(world)

        if shift % training_shifts_before_save == 0:
            policy_world.save_world([world.get_train_directory(), shift])

        policy_world.run()
        world.policy = policy_world.policy
        progress_bar.next()


if __name__ == "__main__":
    SAMPLE_SIZE = 2500
    NUMBER_OF_CLUSTERS = 50

    POLICY = decision.RolloutValueFunctionPolicy(
        decision.EpsilonGreedyValueFunctionPolicy(
            decision.value_functions.LinearValueFunction()
        )
    )

    world_to_analyse = classes.World(
        globals.SHIFT_DURATION,
        None,
        clustering.scripts.get_initial_state(SAMPLE_SIZE, NUMBER_OF_CLUSTERS),
        verbose=False,
        visualize=False,
    )
    world_to_analyse.policy = world_to_analyse.set_policy(POLICY)
    train_value_function(world_to_analyse)
