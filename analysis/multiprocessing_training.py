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
        WEIGHT_UPDATE_STEP_SIZE=learning_rate,
        ANN_NETWORK_STRUCTURE=[100, 100, 100, 100],
    )
    world.policy = world.set_policy(
        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
        value_function_class=decision.value_functions.ANNValueFunction,
    )
    train_value_function(world, save_suffix=f"{suffix}")


def run_learning_rates(input_arg):
    learning_rates(input_arg, f"learn_r_{input_arg}")


def multiprocess_train(inputs, function):
    with Pool() as p:
        p.map(function, inputs)


if __name__ == "__main__":
    multiprocess_train([0.001, 0.0001, 0.00001, 0.000001], run_learning_rates)
