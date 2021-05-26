import copy
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
        if epsilon_decay and shift > 0:
            policy_world.policy.epsilon -= (
                world.INITIAL_EPSILON - world.FINAL_EPSILON
            ) / number_of_shifts
        if shift % world.TRAINING_SHIFTS_BEFORE_SAVE == 0:
            policy_world.save_world(
                cache_directory=world.get_train_directory(save_suffix), suffix=shift
            )

        policy_world.policy.value_function.increment_shifts_trained()

        # avoid running the world after the last model is saved
        if shift != number_of_shifts:
            if scenario_training:
                training_simulation.scripts.training_simulation(policy_world)
            else:
                policy_world.run()

            world.policy = policy_world.policy
            progress_bar.next()


if __name__ == "__main__":
    import classes
    import clustering.scripts
    import decision.value_functions
    import sys
    import os
    from pyinstrument import Profiler

    if len(sys.argv) > 1:
        path = sys.argv[1]
        world_obj_path = path.split("/")[-1]
        dir_path = path.replace(world_obj_path, "")
        world_to_train = classes.World.load(os.path.join(dir_path, world_obj_path))
        world_to_train.policy.value_function.model.reset_tensorboard(
            world_to_train.ANN_NETWORK_STRUCTURE
        )

    else:
        SAMPLE_SIZE = 2500
        NUMBER_OF_CLUSTERS = 50
        standard_parameters = globals.HyperParameters()
        world_to_train = classes.World(
            960,
            None,
            clustering.scripts.get_initial_state(
                SAMPLE_SIZE,
                NUMBER_OF_CLUSTERS,
                number_of_vans=4,
                number_of_bikes=0,
            ),
            verbose=False,
            visualize=False,
            MODELS_TO_BE_SAVED=1,
            TRAINING_SHIFTS_BEFORE_SAVE=1,
            ANN_NETWORK_STRUCTURE=[3000, 2000, 1000, 500, 250, 175, 100, 50],
            REPLAY_BUFFER_SIZE=300,
        )
        world_to_train.policy = world_to_train.set_policy(
            policy_class=decision.EpsilonGreedyValueFunctionPolicy,
            value_function_class=decision.value_functions.ANNValueFunction,
        )

    profiler = Profiler()
    profiler.start()

    train_value_function(world_to_train)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
