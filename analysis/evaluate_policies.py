import copy
import os

import classes
import clustering.scripts
import decision.value_functions
from visualization.visualizer import visualize_analysis


def run_analysis_from_path(path: str):
    world_objects = [
        classes.World.load(os.path.join(path, world_obj_path))
        for world_obj_path in os.listdir(path)
    ]
    initial_state_world, *rest = world_objects
    initial_state_world.visualize = True
    initial_state_world.shift_duration = 480
    policies = [world.policy for world in world_objects]
    return run_analysis(policies, initial_state_world)


def run_analysis(policies, world: classes.World):
    instances = []
    for current_policy in policies:
        print(
            f"\n---------- {current_policy} - {current_policy.value_function.shifts_trained} ----------"
        )
        policy_world = copy.deepcopy(world)
        policy_world.policy = policy_world.set_policy(current_policy)
        # run the world and add the world object to a list containing all world instances
        policy_world.run()
        instances.append(policy_world)

    # visualize policy analysis
    if world.visualize:
        visualize_analysis(instances)
    return instances


def example_setup():
    SHIFT_DURATION = 80
    SAMPLE_SIZE = 100
    NUMBER_OF_CLUSTERS = 10

    # different policies: RandomRolloutPolicy, SwapAllPolicy, TD0Policy
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
        "world_cache/trained_models/LinearValueFunction/c50_s2500/2021-04-27T18:09"
    )
