import copy
import globals
import classes
import clustering.scripts
import decision.value_functions
from progress.bar import IncrementalBar


def train_value_function(world):
    progress_bar = IncrementalBar(
        "Running World",
        check_tty=False,
        max=(globals.TRAINING_SHIFTS_BEFORE_SAVE * globals.MODELS_TO_BE_SAVED),
        suffix="%(percent)d%% - ETA %(eta)ds",
    )
    print(
        f"-------------------- {world.policy.roll_out_policy.value_function} training --------------------"
    )
    for shift in range(
        globals.TRAINING_SHIFTS_BEFORE_SAVE * globals.MODELS_TO_BE_SAVED
    ):
        policy_world = copy.deepcopy(world)

        if shift % globals.TRAINING_SHIFTS_BEFORE_SAVE == 0:
            time_stamp = policy_world.created_at.split("_")[0]
            training_directory = (
                f"{policy_world.policy.roll_out_policy.value_function}/"
                f"c{len(world.state.clusters)}_s{len(world.state.get_scooters())}/{time_stamp}"
            )
            policy_world.save_world([training_directory, shift])

        policy_world.run()
        world.policy = policy_world.policy
        progress_bar.next()


if __name__ == "__main__":
    SAMPLE_SIZE = 100
    NUMBER_OF_CLUSTERS = 10

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
