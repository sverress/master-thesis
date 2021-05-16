import copy
import os
import math
from multiprocessing import Pool
import classes
import decision.value_functions
import decision
import analysis.export_metrics_to_xlsx
from visualization.visualizer import visualize_analysis


def run_analysis_from_path(
    path: str,
    visualize_route=False,
    runs_per_policy=10,
    shift_duration=960,
    world_attribute="SHIFT_DURATION",
    number_of_extra_vehicles=0,
    export_to_excel=False,
    multiprocess=True,
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

        # Do not learn anything more when during evaluation
        world.disable_training = True

        # Add extra vans
        for _ in range(number_of_extra_vehicles):
            world.add_van()

    return run_analysis(
        world_objects,
        runs_per_policy=runs_per_policy,
        title=path.split("/")[-1],  # Use "last" folder as title in the plot
        world_attribute=world_attribute,
        export_to_excel=export_to_excel,
        multiprocess=multiprocess,
    )


def evaluate_world(world, world_attribute, verbose, runs_per_policy):
    world.label = (
        f"{world.policy} - {world_attribute}: {getattr(world, world_attribute)}"
    )
    if verbose:
        print(f"\n---------- {world.label} ----------")
    metrics = []
    for _ in range(runs_per_policy):
        run_world = copy.deepcopy(world)
        # run the world and add the world object to a list containing all world instances
        run_world.run()
        metrics.append(run_world.metrics)
    world.metrics = classes.World.WorldMetric.aggregate_metrics(metrics)

    td_error_tuple = None
    if hasattr(world.policy, "value_function"):
        td_error_tuple = (
            world.policy.value_function.td_errors,
            world.policy.__str__(),
        )
    return world, td_error_tuple


def run_analysis(
    worlds,
    runs_per_policy=4,
    verbose=True,
    save=False,
    baseline_policy_world=None,
    title=None,
    world_attribute="SHIFT_DURATION",
    export_to_excel=False,
    multiprocess=True,
):
    instances = []
    if baseline_policy_world:
        first_world = baseline_policy_world
    else:
        first_world, *rest = worlds
    # Always add a policy that does nothing and a random action
    for baseline_policy_class in [
        decision.DoNothing,
        decision.RebalancingPolicy,
        decision.SwapAllPolicy,
    ]:
        # Use the first world as the world for baseline policies
        baseline_policy_world = copy.deepcopy(first_world)
        baseline_policy_world.policy = baseline_policy_world.set_policy(
            policy_class=baseline_policy_class
        )
        worlds.append(baseline_policy_world)

    td_errors_and_label = []

    if multiprocess:
        with Pool() as p:
            results = p.starmap(
                evaluate_world,
                [
                    (world, world_attribute, verbose, runs_per_policy)
                    for world in worlds
                ],
            )
            for world_result, td_error_tuple_result in results:
                td_errors_and_label.append(td_error_tuple_result)
                instances.append(world_result)
    else:
        for world in worlds:
            world_result, td_error_tuple_result = evaluate_world(
                world, world_attribute, verbose, runs_per_policy
            )
            td_errors_and_label.append(td_error_tuple_result)
            instances.append(world_result)

    visualize_analysis(instances, title=title)
    if save:
        for world_result in instances:
            world_result.save_world()

    if export_to_excel:
        analysis.export_metrics_to_xlsx.metrics_to_xlsx(instances)

    return instances


if __name__ == "__main__":
    import sys

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if len(sys.argv) > 1:
        print(f"fetching world objects from {sys.argv[2]}")
        run_analysis_from_path(sys.argv[2], world_attribute=sys.argv[1])
    else:
        run_analysis_from_path(
            "world_cache/test_models",
            shift_duration=80,
            runs_per_policy=1,
            visualize_route=True,
        )
