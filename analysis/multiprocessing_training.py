from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def main_setup(shift_duration, suffix):
    world = classes.World(
        shift_duration,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )
    train_value_function(world, save_suffix=f"{suffix}")


def run_train_with_shift_duration(input_arg):
    main_setup(input_arg, f"shift_{input_arg}")


def multiprocess_train(inputs, function):
    with Pool() as p:
        p.map(function, inputs)


if __name__ == "__main__":
    multiprocess_train([1, 2, 3], run_train_with_shift_duration)
