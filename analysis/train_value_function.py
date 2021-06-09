import copy
import time

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
    training_times = []
    for shift in range(number_of_shifts + 1):
        start = time.time()
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

        # avoid running the world after the last model is saved
        if shift != number_of_shifts:
            if scenario_training:
                training_simulation.scripts.training_simulation(policy_world)
            else:
                policy_world.run()

            world.policy = policy_world.policy
            progress_bar.next()
        training_times.append(time.time() - start)

    return sum(training_times) / number_of_shifts


if __name__ == "__main__":
    import pandas as pd
    import os

    SAMPLE_SIZE = 2500
    NUMBER_OF_CLUSTERS = [10, 20, 30, 50, 75, 100, 200, 300]
    standard_parameters = globals.HyperParameters()
    decision_times = []
    for num_clusters in NUMBER_OF_CLUSTERS:
        world_to_analyse = classes.World(
            960,
            None,
            clustering.scripts.get_initial_state(
                SAMPLE_SIZE,
                num_clusters,
                number_of_vans=2,
                number_of_bikes=0,
            ),
            verbose=False,
            visualize=False,
            MODELS_TO_BE_SAVED=1,
            TRAINING_SHIFTS_BEFORE_SAVE=10,
            ANN_LEARNING_RATE=0.0001,
            ANN_NETWORK_STRUCTURE=[1000, 2000, 1000, 200],
            REPLAY_BUFFER_SIZE=64,
        )
        world_to_analyse.policy = world_to_analyse.set_policy(
            policy_class=decision.EpsilonGreedyValueFunctionPolicy,
            value_function_class=decision.value_functions.ANNValueFunction,
        )
        for cluster in world_to_analyse.state.clusters:
            cluster.scooters = cluster.scooters[: round(len(cluster.scooters) * 0.6)]
            cluster.ideal_state = round(cluster.ideal_state * 0.6)

        decision_times.append(train_value_function(world_to_analyse))

    df = pd.DataFrame(
        decision_times,
        index=NUMBER_OF_CLUSTERS,
        columns=["Avg. time per shift"],
    )

    if not os.path.exists("computational_study"):
        os.makedirs("computational_study")

    df.to_excel("computational_study/training_time_clusters_shift_short.xlsx")
