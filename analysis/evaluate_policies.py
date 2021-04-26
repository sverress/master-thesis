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
    policies = [world.policy.roll_out_policy for world in world_objects]
    return run_analysis(policies, initial_state_world)


def run_analysis(policies, world: classes.World):
    instances = []
    for current_policy in policies:
        print(f"\n---------- {current_policy} ----------")
        policy_world = copy.deepcopy(world)
        policy_world.policy = policy_world.set_policy(current_policy)
        # run the world and add the world object to a list containing all world instances
        policy_world.run()
        instances.append(policy_world)

    # visualize policy analysis
    if world.visualize:
        visualize_analysis(instances, policies)
    return instances


if __name__ == "__main__":
    SHIFT_DURATION = 60
    SAMPLE_SIZE = 100
    NUMBER_OF_CLUSTERS = 10

    # different policies: RandomRolloutPolicy, SwapAllPolicy, TD0Policy
    POLICIES = [
        decision.RolloutValueFunctionPolicy(
            decision.EpsilonGreedyValueFunctionPolicy(
                decision.value_functions.LinearValueFunction()
            )
        ),
        decision.RandomRolloutPolicy(),
    ]
    WORLD = classes.World(
        SHIFT_DURATION,
        None,
        clustering.scripts.get_initial_state(SAMPLE_SIZE, NUMBER_OF_CLUSTERS),
    )
    run_analysis(POLICIES, WORLD)
