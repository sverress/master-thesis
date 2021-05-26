import os
from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def learning_rates(input_arguments, suffix):
    world = classes.World(
        960,
        None,
        clustering.scripts.get_initial_state(2500, 50),
        verbose=False,
        visualize=False,
        test_parameter_name="learning_rate",
        test_parameter_value=input_arguments,
        ANN_LEARNING_RATE=input_arguments,
        ANN_NETWORK_STRUCTURE=[3000, 3000, 3000, 2000, 1000, 500, 250, 175, 100, 50],
        TRAINING_SHIFTS_BEFORE_SAVE=1_000,
        MODELS_TO_BE_SAVED=10,
        REPLAY_BUFFER_SIZE=500,
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

    multiprocess_train(
        learning_rates,
        [
            (value, f"kombinasjon_{value}")
            for value in [0.0001, 0.00001, 0.01, 0.001, 0.000001]
        ],
    )
