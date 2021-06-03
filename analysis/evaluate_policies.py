import copy
import os
import math
from multiprocessing import Pool
import classes
import decision.value_functions
import decision
import analysis.export_metrics_to_xlsx
import system_simulation.scripts
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
    return_worlds=False,
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

        world.policy.epsilon = 0

        for vehicle in world.state.vehicles:
            vehicle.service_route = []

        # Add extra vans
        for _ in range(number_of_extra_vehicles):
            world.add_van()

    if return_worlds:
        return world_objects
    else:
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
    # for baseline_policy_class in [
    #    decision.DoNothing,
    ##    decision.SwapAllPolicy,
    # ]:
    #    # Use the first world as the world for baseline policies
    #    baseline_policy_world = copy.deepcopy(first_world)
    #    baseline_policy_world.policy = baseline_policy_world.set_policy(
    #        policy_class=baseline_policy_class
    #    )
    # worlds.append(baseline_policy_world)

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

    # visualize_analysis(instances, title=title)
    if save:
        for world_result in instances:
            world_result.save_world()

    if export_to_excel:
        analysis.export_metrics_to_xlsx.metrics_to_xlsx(instances)

    return instances


if __name__ == "__main__":
    import sys
    import clustering.scripts
    import clustering.methods
    import pandas as pd

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    if len(sys.argv) > 1:
        print(f"fetching world objects from {sys.argv[2]}")
        run_analysis_from_path(
            sys.argv[2], world_attribute=sys.argv[1], runs_per_policy=3
        )
    else:

        """

        instances = run_analysis_from_path(
            "world_cache/trained_models/ANNValueFunction/c50_s1998/longest_trained",
            shift_duration=960,
            runs_per_policy=10,
        )
        """

        number_of_scooters = [1500]
        SAMPLE_SIZE = 2500
        NUMBER_OF_CLUSTERS = [10, 20, 30, 50, 75, 100, 200, 400, 500]
        divides = [2, 3, 4, 5, 10]
        all_times = []
        instances = []

        try:
            for clusters in NUMBER_OF_CLUSTERS:
                cluster_times = []
                state = clustering.scripts.get_initial_state(
                    SAMPLE_SIZE,
                    clusters,
                    number_of_vans=2,
                    number_of_bikes=0,
                )

                # system simulate the states to shake up the states
                for i in range(5):
                    system_simulation.scripts.system_simulate(state)

                sample_size = number_of_scooters[0]

                percentage = sample_size / SAMPLE_SIZE
                for cluster in state.clusters:
                    cluster.scooters = cluster.scooters[
                        : round(len(cluster.scooters) * percentage)
                    ]
                    cluster.ideal_state = round(cluster.ideal_state * percentage)

                world_to_analyse = classes.World(
                    960,
                    None,
                    state,
                    verbose=False,
                    visualize=False,
                    MODELS_TO_BE_SAVED=5,
                    TRAINING_SHIFTS_BEFORE_SAVE=50,
                    ANN_LEARNING_RATE=0.0001,
                    ANN_NETWORK_STRUCTURE=[1000, 2000, 200],
                    REPLAY_BUFFER_SIZE=100,
                    test_parameter_name="divide",
                )

                worlds = []
                for divide in divides:
                    world = copy.deepcopy(world_to_analyse)
                    world.policy = world.set_policy(
                        policy_class=decision.EpsilonGreedyValueFunctionPolicy,
                        value_function_class=decision.value_functions.LinearValueFunction,
                    )
                    world.disable_training = True
                    world.policy.epsilon = 0
                    world.DIVIDE_GET_POSSIBLE_ACTIONS = divide
                    worlds.append(world)

                instances = run_analysis(
                    worlds,
                    baseline_policy_world=world_to_analyse,
                    runs_per_policy=1,
                )
                time = []
                number_of_actions = []

                for instance in instances:
                    time += instance.policy.time
                    number_of_actions += instance.policy.number_of_actions

                    print(
                        f"Number of clusters - {clusters} | Divide - {instance.NUMBER_OF_NEIGHBOURS}\n"
                        f"Avg decision time: {sum(time) / len(time)}\n"
                        f"Avg number of actions: {sum(number_of_actions) / len(number_of_actions)}"
                        f" | Number of decisions: {len(number_of_actions)}\n"
                        f"Actions: {number_of_actions}"
                    )
                    cluster_times.append(sum(time) / len(time))
                all_times.append(cluster_times)
        finally:
            df = pd.DataFrame(all_times, index=NUMBER_OF_CLUSTERS, columns=divides)

            df.to_excel("computational_study/tech_analysis.xlsx")
