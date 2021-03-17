import classes
import matplotlib.pyplot as plt
import numpy as np
from globals import ITERATION_LENGTH_MINUTES, COLORS


def run_analysis():
    shift_duration = 50
    policies = [
        "RandomRolloutPolicy",
        "SwapAllPolicy",
    ]
    instances = []
    for policy in policies:
        world = classes.World(
            shift_duration, sample_size=100, number_of_clusters=10, policy=policy
        )
        world.stack.append(classes.GenerateScooterTrips(0))
        world.stack.append(classes.VehicleArrival(0, world.state.current_cluster.id))
        world.run()
        instances.append(world)

    x = np.arange(0, shift_duration + 1, ITERATION_LENGTH_MINUTES)

    plots = []
    for i, instance in enumerate(instances):
        (
            lost_demand,
            deviation_ideal_state,
            deficient_battery,
        ) = instance.metrics.get_all_metrics()
        (demand_plot,) = plt.plot(
            x, np.array(lost_demand).T, linestyle="-", c=COLORS[i]
        )
        (deviation_plot,) = plt.plot(
            x, np.array(deviation_ideal_state).T, linestyle="--", c=COLORS[i]
        )
        (deficient_plot,) = plt.plot(
            x, np.array(deficient_battery).T, linestyle=":", c=COLORS[i]
        )

        plots.append([demand_plot, deviation_plot, deficient_plot])

    x_position = round(len(x) / 2) * ITERATION_LENGTH_MINUTES

    legend1 = plt.legend(
        plots[0],
        ["Lost demand", "Deviation IS", "Deficient battery"],
        bbox_to_anchor=(x_position, 1.05),
    )
    plt.legend(
        [plot[0] for plot in plots], policies, bbox_to_anchor=(x_position, -0.05)
    )
    plt.gca().add_artist(legend1)

    plt.show()


run_analysis()
