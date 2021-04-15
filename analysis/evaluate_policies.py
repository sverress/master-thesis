import classes
import decision
import decision.value_functions
from visualization.visualizer import visualize_analysis


def run_analysis(
    shift_duration=100,
    sample_size=100,
    number_of_clusters=10,
    policy=None,
    visualize_world=True,
    verbose=False,
):
    """
    Method to run different policies and analysis their performance
    :param visualize_world:
    :param policy:
    :param verbose: show verbose in console
    :param shift_duration: total shift to be analysed
    :param sample_size: size of instances
    :param number_of_clusters: number of clusters in the world
    :return: matplotlib figure - figure containing plot of the analysis
    """

    # create the world object with given input parameters
    world = classes.World(
        shift_duration,
        sample_size=sample_size,
        number_of_clusters=number_of_clusters,
        policy=policy,
        visualize=visualize_world,
        verbose=verbose,
    )
    # pumping up the trip intensity
    for cluster in world.state.clusters:
        cluster.trip_intensity_per_iteration = round(cluster.ideal_state * 0.1)
    # run the world and add the world object to a list containing all world instances
    world.run()

    return world


if __name__ == "__main__":
    SHIFT_DURATION = 120
    SAMPLE_SIZE = 100
    NUMBER_OF_CLUSTERS = 10
    NUMBER_OF_DEPOTS = 3

    # different value functions: GradientDescent
    VALUE_FUNCTION = decision.value_functions.LinearValueFunction(
        number_of_locations=NUMBER_OF_CLUSTERS + NUMBER_OF_DEPOTS,
        number_of_clusters=NUMBER_OF_CLUSTERS,
    )
    ROLL_OUT_POLICY = decision.EpsilonGreedyValueFunctionPolicy(VALUE_FUNCTION)
    # different policies: RandomRolloutPolicy, SwapAllPolicy, TD0Policy
    POLICIES = [
        decision.ValueFunctionPolicy(ROLL_OUT_POLICY),
        decision.RandomRolloutPolicy(),
    ]

    instances = []
    for current_policy in POLICIES:
        print(f"\n---------- {current_policy.__str__()} ----------")
        policy_world = run_analysis(
            shift_duration=SHIFT_DURATION,
            sample_size=SAMPLE_SIZE,
            number_of_clusters=NUMBER_OF_CLUSTERS,
            policy=current_policy,
            visualize_world=True,
            verbose=True,
        )
        instances.append(policy_world)

    # visualize the world instances that have been run
    figure = visualize_analysis(instances, POLICIES, smooth_curve=True)
