from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def discount_rates(discount_rate, suffix):
    world = classes.World(
        960,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        test_parameter_name="discount_rate",
        test_parameter_value=discount_rate,
        DISCOUNT_RATE=discount_rate,
        ANN_NETWORK_STRUCTURE=[100, 100, 100, 100],
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
        discount_rates,
        [
            (value, f"lr_{value}")
            for value in [0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.999, 0.9999]
        ],
    )
