from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import globals
import clustering.scripts


def main_setup(value_function, suffix):
    world = classes.World(
        globals.SHIFT_DURATION,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
    )
    world.policy = world.set_policy(
        decision.EpsilonGreedyValueFunctionPolicy(value_function)
    )
    train_value_function(world, save_suffix=f"{suffix}")


def run_train_with_shift_duration(input_arg):
    globals.SHIFT_DURATION = input_arg
    globals.TRAINING_SHIFTS_BEFORE_SAVE = 1
    value_function = decision.value_functions.ANNValueFunction([100, 100])
    main_setup(value_function, f"shift_{input_arg}")


def multiprocess_train(inputs, function):
    with Pool() as p:
        p.map(function, inputs)


if __name__ == "__main__":
    multiprocess_train([1, 2, 3], run_train_with_shift_duration)
