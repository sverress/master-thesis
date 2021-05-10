from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def learning_rates(learning_rate, suffix):
    world = classes.World(
        960,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        test_parameter_name="learning_rate",
        test_parameter_value=learning_rate,
        WEIGHT_UPDATE_STEP_SIZE=learning_rate,
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.LinearValueFunction,
    )
    train_value_function(world, save_suffix=f"{suffix}")


def multiprocess_train(function, inputs):
    with Pool() as p:
        p.starmap(function, inputs)


if __name__ == "__main__":
    multiprocess_train(
        learning_rates,
        [
            (value, f"lr_{value}")
            for value in [
                0.01,
                0.005,
                0.001,
                0.0005,
                0.0001,
                0.0005,
                0.00001,
                0.00005,
                0.000001,
                0.000005,
            ]
        ],
    )
