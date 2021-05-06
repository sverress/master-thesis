from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def learning_rates_ann(learning_rate, suffix):
    world = classes.World(
        480,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        WEIGHT_UPDATE_STEP_SIZE=learning_rate,
        ANN_NETWORK_STRUCTURE=[100, 100, 100, 100],
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )
    train_value_function(world, save_suffix=f"{suffix}")


def learning_rates_linear(learning_rate, suffix):
    world = classes.World(
        480,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        WEIGHT_UPDATE_STEP_SIZE=learning_rate,
        LOCATION_REPETITION=1,
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.LinearValueFunction,
    )
    train_value_function(world, save_suffix=f"{suffix}")


def value_function_is_learning(value_function, suffix):
    world = classes.World(
        480,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        LOCATION_REPETITION=3,
        NUMBER_OF_NEIGHBOURS=15,
        MODELS_TO_BE_SAVED=5,
        TRAINING_SHIFTS_BEFORE_SAVE=50,
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=value_function,
    )
    train_value_function(world, save_suffix=f"{suffix}")


def multiprocess_train(function, inputs):
    with Pool() as p:
        p.starmap(function, inputs)


if __name__ == "__main__":
    multiprocess_train(
        learning_rates,
        [(value, f"lr_{value}") for value in [0.001, 0.0001, 0.00001, 0.000001]],
    )
