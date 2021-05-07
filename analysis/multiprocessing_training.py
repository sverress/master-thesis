from multiprocessing import Pool

import decision.value_functions
import classes
from analysis.train_value_function import train_value_function
import clustering.scripts


def lost_trip_rewards(lost_trip_reward, suffix):
    world = classes.World(
        480,
        None,
        clustering.scripts.get_initial_state(2500, 30),
        verbose=False,
        visualize=False,
        test_parameter_name="lost_trip_reward",
        test_parameter_value=lost_trip_reward,
        LOST_TRIP_REWARD=lost_trip_reward,
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
        lost_trip_rewards,
        [
            (value, f"lr_{value}")
            for value in [-0.01, -0.1, -1, -5, -10, -50, -100, -1000]
        ],
    )
