import sys
from instance.InstanceManager import InstanceManager
from instance.helpers import save_models_to_excel
import matplotlib.pyplot as plt
import numpy as np


def run_from_json():
    manager = InstanceManager()
    manager.create_multiple_instances()
    for instance_key in manager.instances.keys():
        instance = manager.get_instance(instance_key)
        instance.run()
        instance.visualize_solution()
        instance.save_model()
    save_models_to_excel()


def plot_function():
    rate = 0.80
    x = np.arange(start=0, stop=10, step=1)
    y = x * rate ** x
    plt.scatter(x, y, s=4, c="black")
    plt.xticks(x)
    for i in x:
        plt.annotate(f" ({i}, {round(y[i], 2)})", (i, y[i]))
    plt.show()


if __name__ == "__main__":
    program = sys.argv[1]
    if program == "run":
        run_from_json()
    if program == "plot":
        plot_function()
