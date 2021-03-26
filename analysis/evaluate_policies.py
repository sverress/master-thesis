import classes
from visualization.visualizer import visualize_analysis


def run_analysis(
    shift_duration=100,
    sample_size=100,
    number_of_clusters=10,
    policies=None,
    visualize_world=False,
    smooth_curve=True,
):
    """
    Method to run different policies and analysis their performance
    :param shift_duration: total shift to be analysed
    :param sample_size: size of instances
    :param number_of_clusters: number of clusters in the world
    :param policies: different policies to be analysed
    :param visualize_world: boolean - if the running of the world should be visualized
    :param smooth_curve: boolean - if the analysed metrics is to be smoothed out in the analysis plot
    :return: matplotlib figure - figure containing plot of the analysis
    """
    instances = []
    # loop over all policies to be analysed - default RandomRolloutPolicy if no policy is given
    for policy in policies if policies else ["RandomRolloutPolicy"]:
        print(f"\n---------- {policy} ----------")
        # create the world object with given input parameters
        world = classes.World(
            shift_duration,
            sample_size=sample_size,
            number_of_clusters=number_of_clusters,
            policy=policy,
        )
        # run the world and add the world object to a list containing all world instances
        world.run()
        instances.append(world)

    # visualize the world instances that have been run
    figure = visualize_analysis(instances, policies, smooth_curve)

    return figure
