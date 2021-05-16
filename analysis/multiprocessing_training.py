import os
from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def learning_rates(input_arguments, suffix):
    learning_rate, ann_structure = input_arguments
    world = classes.World(
        960,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        test_parameter_name="learning_rate",
        test_parameter_value=learning_rate,
        WEIGHT_UPDATE_STEP_SIZE=learning_rate,
        ANN_NETWORK_STRUCTURE=ann_structure,
        TRAINING_SHIFTS_BEFORE_SAVE=100,
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )
    train_value_function(world, save_suffix=f"{suffix}")


def multiprocess_train(function, inputs):
    with Pool() as p:
        p.starmap(function, inputs)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import itertools

    multiprocess_train(
        learning_rates,
        [
            (value, f"kombinasjon_{value}")
            for value in list(
                itertools.product(
                    [0.0001, 0.00001, 0.001],
                    [
                        [
                            1000,
                            1000,
                            500,
                            100,
                            500,
                            1000,
                            1000,
                            3000,
                            1000,
                            1000,
                            500,
                            100,
                        ],
                        [1000, 1000, 500, 100, 500, 1000, 3000, 1000, 500, 100],
                        [1000, 1000, 500, 100, 500, 1000, 500, 100],
                    ],
                )
            )
        ],
    )
