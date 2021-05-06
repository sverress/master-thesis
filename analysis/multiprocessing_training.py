from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def train_linear_value_function(learning_rate, suffix):
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


def run_train_with_learning_rate(input_arg):
    train_linear_value_function(input_arg, f"lr_{input_arg}")


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


def run_value_function_is_learning(input_arg):
    value_function_is_learning(input_arg, f"func_{input_arg}")


def multiprocess_train(inputs, function):
    with Pool() as p:
        p.map(function, inputs)


if __name__ == "__main__":
    multiprocess_train(
        [
            decision.value_functions.ANNValueFunction,
            decision.value_functions.LinearValueFunction,
        ],
        run_value_function_is_learning,
    )
