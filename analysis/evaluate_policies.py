import copy

import classes
import clustering.scripts
import decision.value_functions
from visualization.visualizer import visualize_analysis


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
    SHIFT_DURATION = 120
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
