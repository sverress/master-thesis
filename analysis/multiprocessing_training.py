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


def train_ann_network_structure(ann_structure, suffix):
    world = classes.World(
        480,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        ANN_NETWORK_STRUCTURE=ann_structure,
        TRAINING_SHIFTS_BEFORE_SAVE=50,
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )
    train_value_function(world, save_suffix=f"{suffix}")


def run_train_ann_network_structure(input_arg):
    train_ann_network_structure(input_arg, f"struct_{input_arg}")


def multiprocess_train(inputs, function):
    with Pool() as p:
        p.map(function, inputs)


if __name__ == "__main__":
    multiprocess_train(
        [[100], [100] * 3, [100] * 5, [100] * 10], run_train_ann_network_structure
    )
