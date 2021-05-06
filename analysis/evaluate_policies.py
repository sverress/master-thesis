import copy
import os
import math
import classes
import decision.value_functions
import decision
from visualization.visualizer import visualize_analysis, visualize_td_error


def run_analysis_from_path(
    path: str,
    visualize_route=False,
    runs_per_policy=4,
    shift_duration=480,
):
    # Sort the policies by the training duration
    world_objects = sorted(
        [
            classes.World.load(os.path.join(path, world_obj_path))
            for world_obj_path in os.listdir(path)
            if world_obj_path != ".DS_Store"
        ],
        key=lambda world_object: world_object.policy.value_function.shifts_trained
        if hasattr(world_object.policy, "value_function")
        else math.inf,
    )

    for world in world_objects:
        # route visualization
        if visualize_route:
            for event in world.stack:
                if isinstance(event, classes.VehicleArrival):
                    event.visualize = True

        # Always rollout for 8 hours
        world.shift_duration = shift_duration

        # Turn off neighbor filtering
        world.NUMBER_OF_NEIGHBOURS = 0

    return run_analysis(
        world_objects,
        runs_per_policy=runs_per_policy,
    )


def run_analysis(
    worlds,
    runs_per_policy=4,
    verbose=True,
    save=False,
    baseline_policy_world=None,
):
    instances = []
    if baseline_policy_world:
        first_world = baseline_policy_world
    else:
        first_world, *rest = worlds
    # Always add a policy that does nothing and a random action
    for baseline_policy_class in [
        decision.DoNothing,
        decision.RandomActionPolicy,
        decision.SwapAllPolicy,
    ]:
        # Use the first world as the world for baseline policies
        baseline_policy_world = copy.deepcopy(first_world)
        baseline_policy_world.policy = baseline_policy_world.set_policy(
            policy_class=baseline_policy_class
        )
        worlds.append(baseline_policy_world)

    td_errors_and_label = []
    for world in worlds:
        if verbose:
            print(f"\n---------- {world.policy} ----------")
        metrics = []
        for _ in range(runs_per_policy):
            run_world = copy.deepcopy(world)
            # run the world and add the world object to a list containing all world instances
            run_world.run()
            metrics.append(run_world.metrics)
        world.metrics = classes.World.WorldMetric.aggregate_metrics(metrics)
        instances.append(world)

        if hasattr(world.policy, "value_function"):
            td_errors_and_label.append(
                (
                    world.policy.value_function.td_errors,
                    world.policy.__str__(),
                )
            )

    visualize_analysis(instances)
    visualize_td_error(td_errors_and_label)
    if save:
        for world in instances:
            world.save_world(cache_directory="evaluated_models")
    return instances


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(f"fetching world objects from {sys.argv[1]}")
        run_analysis_from_path(sys.argv[1])
    else:
        run_analysis_from_path(
            "world_cache/trained_models/LinearValueFunction/c30_s2500/TEST_SET"
        )
