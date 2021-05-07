from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def learning_rates(learning_rate, suffix):
    world = classes.World(
        480,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        test_parameter_name="learning_rate",
        test_parameter_value=learning_rate,
        WEIGHT_UPDATE_STEP_SIZE=learning_rate,
        ANN_NETWORK_STRUCTURE=[100, 100, 100, 100],
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )
    train_value_function(world, save_suffix=f"{suffix}")


def ann_structure(structure, suffix):
    world = classes.World(
        480,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        test_parameter_name="structure",
        test_parameter_value=structure,
        ANN_NETWORK_STRUCTURE=structure,
        TRAINING_SHIFTS_BEFORE_SAVE=10,
        MODELS_TO_BE_SAVED=3,
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
    multiprocess_train(
        learning_rates,
        [
            (value, f"structure_{value}")
            for value in [
                [30] * 10,
                [30] * 5,
                [100],
                [100] * 3,
                [100] * 6,
                [1000],
                [1000] * 3,
                [100, 50, 10],
                [100, 50, 10, 5, 2],
                [1000, 500, 200, 100],
                [1000, 500, 200, 100, 50, 20, 10],
            ]
        ],
    )
