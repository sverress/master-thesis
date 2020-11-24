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

    beta = 0.8
    x = np.arange(start=1, stop=9, step=1)
    ideal_state = 8
    sum_b = 4
    theta = 0.05

    y = []
    for i in x:
        y.append(calc_r_kz(0, beta, theta, ideal_state, sum_b, i) if i == 1
                 else calc_r_kz(y[-1], beta, theta, ideal_state, sum_b, i))

    plt.scatter(x, y, s=4, c="black")
    plt.xticks([i for i in range(0, len(x)+2)])
    plt.xlabel("k")
    plt.ylabel("R_kz")
    plt.xlim(0, 10)
    plt.ylim(0, 8)
    for i in x:
        plt.annotate(f" ({i}, {round(y[i-1], 2)})", (i, y[i-1]))
    plt.show()


def calc_r_kz(value, beta, theta, ideal_state, sum_b, x):
    return value + beta + theta*((ideal_state-sum_b)/x)


if __name__ == "__main__":
    program = sys.argv[1]
    if program == "run":
        run_from_json()
    if program == "plot":
        plot_function()
