import copy
import os
import math
import classes
import clustering.scripts
import decision.value_functions
import decision
import globals
from visualization.visualizer import visualize_analysis


def run_analysis_from_path(path: str, other_policies=None, visualize_route=False):
    world_objects = [
        classes.World.load(os.path.join(path, world_obj_path))
        for world_obj_path in os.listdir(path)
    ]
    initial_state_world, *rest = world_objects

    # this is for analysis visualization
    initial_state_world.visualize = True

    # route visualization
    if visualize_route:
        for event in initial_state_world.stack:
            if isinstance(event, classes.VehicleArrival):
                event.visualize = True

    initial_state_world.shift_duration = 480
    policies = sorted(
        [world.policy for world in world_objects],
        key=lambda policy: policy.value_function.shifts_trained
        if hasattr(policy, "value_function")
        else math.inf,
    )

    return run_analysis(
        policies + other_policies if other_policies else policies, initial_state_world
    )


def run_analysis(policies, world: classes.World, smooth_curve=True):
    instances = []
    for current_policy in policies:
        print(f"\n---------- {current_policy} ----------")

        policy_world = copy.deepcopy(world)
        # Set the number of neighbors to half the number of clusters in the state
        current_policy.number_of_neighbors = len(policy_world.state.clusters) / 2
        policy_world.policy = policy_world.set_policy(current_policy)
        # run the world and add the world object to a list containing all world instances
        policy_world.run()
        print_lost_reward(policy_world.rewards)
        instances.append(policy_world)

    # visualize policy analysis
    if world.visualize:
        visualize_analysis(instances, smooth_curve)
    return instances


def print_lost_reward(rewards):
    print("Lost demand in cluster: ")
    for reward, location_id in rewards:
        if reward == globals.LOST_TRIP_REWARD:
            print(location_id)


def example_setup():
    SHIFT_DURATION = 80
    SAMPLE_SIZE = 100
    NUMBER_OF_CLUSTERS = 10

    POLICIES = [
        decision.EpsilonGreedyValueFunctionPolicy(
            decision.value_functions.LinearValueFunction()
        ),
        decision.SwapAllPolicy(),
    ]
    WORLD = classes.World(
        SHIFT_DURATION,
        None,
        clustering.scripts.get_initial_state(SAMPLE_SIZE, NUMBER_OF_CLUSTERS),
        visualize=False,
    )
    run_analysis(POLICIES, WORLD)


if __name__ == "__main__":
    run_analysis_from_path(
        "world_cache/trained_models/LinearValueFunction/c10_s100/2021-04-28T16:48",
        [decision.SwapAllPolicy(), decision.RandomActionPolicy()],
    )
