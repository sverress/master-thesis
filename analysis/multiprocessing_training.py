import os
from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def learning_rates(input_arguments, suffix):
    SAMPLE_SIZE = 2500
    NUMBER_OF_CLUSTERS = 50
    learning_rate, network_structure = input_arguments
    world_to_analyse = classes.World(
        960,
        None,
        clustering.scripts.get_initial_state(
            SAMPLE_SIZE,
            NUMBER_OF_CLUSTERS,
            number_of_vans=2,
            number_of_bikes=0,
        ),
        verbose=False,
        visualize=False,
        MODELS_TO_BE_SAVED=5,
        TRAINING_SHIFTS_BEFORE_SAVE=50,
        ANN_LEARNING_RATE=learning_rate,
        ANN_NETWORK_STRUCTURE=network_structure,
        REPLAY_BUFFER_SIZE=100,
    )
    world_to_analyse.policy = world_to_analyse.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )
    for cluster in world_to_analyse.state.clusters:
        cluster.scooters = cluster.scooters[: round(len(cluster.scooters) * 0.8)]
        cluster.ideal_state = round(cluster.ideal_state * 0.8)
    train_value_function(world_to_analyse, save_suffix=f"{suffix}")


def multiprocess_train(function, inputs):
    with Pool() as p:
        p.starmap(function, inputs)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import itertools

    multiprocess_train(
        learning_rates,
        [
            (value, f"relu_lr_struct_{value}")
            for value in list(
                itertools.product(
                    [0.0001, 0.00001, 0.001, 0.01],
                    [
                        [1000],
                        [1000] * 2,
                        [1000] * 3,
                        [500],
                        [500] * 2,
                        [500] * 3,
                        [100],
                        [100] * 2,
                        [100] * 3,
                    ],
                )
            )
        ],
    )
