import copy
import os
import math
import classes
import clustering.scripts
import decision.value_functions
import decision
import globals
from visualization.visualizer import visualize_analysis, visualize_td_error


def run_analysis_from_path(
    path: str,
    other_policies=None,
    visualize_route=False,
    runs_per_policy=4,
    shift_duration=480,
):
    world_objects = [
        classes.World.load(os.path.join(path, world_obj_path))
        for world_obj_path in os.listdir(path)
        if world_obj_path != ".DS_Store"
    ]
    initial_state_world, *rest = world_objects

    # this is for analysis visualization
    initial_state_world.visualize = True

    # route visualization
    if visualize_route:
        for event in initial_state_world.stack:
            if isinstance(event, classes.VehicleArrival):
                event.visualize = True

    # Always rollout for 8 hours
    initial_state_world.shift_duration = shift_duration
    policies = sorted(
        [world.policy for world in world_objects],
        key=lambda policy: policy.value_function.shifts_trained
        if hasattr(policy, "value_function")
        else math.inf,
    )

    return run_analysis(
        policies + other_policies if other_policies else policies,
        initial_state_world,
        runs_per_policy=runs_per_policy,
    )


def run_analysis(
    policies,
    world: classes.World,
    smooth_curve=True,
    runs_per_policy=4,
    verbose=True,
):
    instances = []
    # Always add a policy that does nothing and a random action
    policies += [
        decision.DoNothing(),
        decision.RandomActionPolicy(),
        decision.SwapAllPolicy(),
    ]
    td_errors_and_label = []
    for current_policy in policies:
        if verbose:
            print(f"\n---------- {current_policy} ----------")
        policy_world = copy.deepcopy(world)
        # Set the number of neighbors to half the number of clusters in the state
        policy_world.policy = policy_world.set_policy(current_policy)
        metrics = []
        for _ in range(runs_per_policy):
            run_policy_world = copy.deepcopy(policy_world)
            # run the world and add the world object to a list containing all world instances
            run_policy_world.run()
            metrics.append(run_policy_world.metrics)
            if verbose:
                print_lost_reward(run_policy_world.rewards)
        policy_world.metrics = classes.World.WorldMetric.aggregate_metrics(metrics)
        instances.append(policy_world)

        if hasattr(policy_world.policy, "roll_out_policy") and hasattr(
            policy_world.policy.roll_out_policy, "value_function"
        ):
            td_errors_and_label.append(
                (
                    policy_world.policy.roll_out_policy.value_function.td_errors,
                    policy_world.policy.roll_out_policy.__str__(),
                )
            )
        elif hasattr(policy_world.policy, "value_function"):
            td_errors_and_label.append(
                (
                    policy_world.policy.value_function.td_errors,
                    policy_world.policy.__str__(),
                )
            )

    visualize_analysis(instances, smooth_curve)
    visualize_td_error(td_errors_and_label, smooth_curve)
    return instances


def print_lost_reward(rewards):
    print("Lost demand in cluster: ")
    for reward, location_id in rewards:
        if reward == globals.LOST_TRIP_REWARD:
            print(location_id)


def example_setup():
    run_analysis(
        [],
        classes.World(
            480,
            None,
            clustering.scripts.get_initial_state(
                2500,
                30,
            ),
            visualize=False,
            verbose=False,
        ),
        smooth_curve=False,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(f"fetching world objects from {sys.argv[1]}")
        run_analysis_from_path(sys.argv[1])
    else:
        run_analysis_from_path(
            "world_cache/trained_models/LinearValueFunction/c30_s2500/TEST_SET"
        )
