import decision.policies
import classes
import matplotlib.pyplot as plt
import numpy as np


def run_analysis():
    shift_duration = 60
    policies = [
        decision.policies.RandomRolloutPolicy(),
        decision.policies.SwapAllPolicy(),
    ]
    instances = []
    for policy in policies:
        world = classes.World(
            shift_duration, sample_size=100, number_of_clusters=10, policy=policy
        )
        world.run()
        instances.append(world)

    x = np.array(0, shift_duration, 20)

    for instance in instances:
        plt.plot(x, np.array(instance.metrics.get_all_metrics()).T)

    plt.show()
