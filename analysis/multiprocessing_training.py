from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
from globals import *
import clustering.scripts


def run_train_with_input_arg(input_arg):
    SAMPLE_SIZE = 2500
    NUMBER_OF_CLUSTERS = 30

    SHIFT_DURATION = input_arg

    POLICY = decision.EpsilonGreedyValueFunctionPolicy(
        decision.value_functions.ANNValueFunction([100, 100])
    )

    world_to_analyse = classes.World(
        SHIFT_DURATION,
        None,
        clustering.scripts.get_initial_state(
            SAMPLE_SIZE,
            NUMBER_OF_CLUSTERS,
            number_of_vans=NUMBER_OF_VANS,
            number_of_bikes=NUMBER_OF_BIKES,
        ),
        verbose=False,
        visualize=False,
    )
    world_to_analyse.policy = world_to_analyse.set_policy(POLICY)
    train_value_function(world_to_analyse, save_suffix=f"{input_arg}")


def multiprocess_train(inputs):
    with Pool() as p:
        p.map(run_train_with_input_arg, inputs)


if __name__ == "__main__":
    multiprocess_train([80, 90, 100])
