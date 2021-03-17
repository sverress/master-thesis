import classes
from visualization.visualizer import visualize_analysis


def run_analysis(
    shift_duration=100,
    sample_size=100,
    number_of_clusters=10,
    policies=None,
    visualize_world=False,
):
    instances = []
    figures = []
    for policy in policies if policies else ["RandomRolloutPolicy"]:
        world = classes.World(
            shift_duration,
            sample_size=sample_size,
            number_of_clusters=number_of_clusters,
            policy=policy,
        )
        world.stack.append(classes.GenerateScooterTrips(0))
        world.stack.append(
            classes.VehicleArrival(0, world.state.current_cluster.id, visualize_world)
        )
        world.run()
        instances.append(world)

    figures.append(visualize_analysis(instances, policies))

    return figures
