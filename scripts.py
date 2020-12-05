import sys
from instance.InstanceManager import InstanceManager
from instance.helpers import save_models_to_excel
import matplotlib.pyplot as plt
import numpy as np


def run_from_json(run_test=False):
    manager = InstanceManager()
    manager.create_multiple_instances(run_test)
    for instance in manager.instances.values():
        # instance.visualize_raw_data_map(
        #    instance.get_model_name(), False, manager.time_stamp
        # )  # instance name, save, time stamp
        instance.print_instance()
        instance.run()
        instance.visualize_solution(
            False, False, manager.time_stamp
        )  # save, edge_plot, time stamp
        instance.model.print_solution()
        instance.save_model_and_instance(manager.time_stamp)
    save_models_to_excel(manager.time_stamp)


def run_project_test():
    """
    Function to run a test on the project. Recommend to change timelimit on TSP to have a faster test.
    The test should take around 5 minutes (64 tests)
    """
    run_from_json(True)


def plot_function():

    beta = 0.8
    ideal_state = 20
    x = np.arange(start=1, stop=ideal_state, step=1)
    sum_b = 0
    theta = 0.05

    y = []
    for i in x:
        r = (
            calc_r_kz(0, beta, theta, ideal_state, sum_b, i)
            if i == 1
            else calc_r_kz(y[-1], beta, theta, ideal_state, sum_b, i)
        )
        y.append(r if r <= i else i)

    plt.scatter(x, y, s=4, c="black")
    plt.xticks([i for i in range(0, len(x) + 2)])
    plt.xlabel("k")
    plt.ylabel("R_kz")
    plt.xlim(0, len(x) + 1)
    plt.ylim(0, len(y) + 1)
    for i in x:
        plt.annotate(f" ({i}, {round(y[i-1], 2)})", (i, y[i - 1]))
    plt.show()


def calc_r_kz(value, beta, theta, ideal_state, sum_b, x):
    return value + beta + theta * ((ideal_state - sum_b) / x)


if __name__ == "__main__":
    program = sys.argv[1]
    if program == "run":
        run_from_json()
    if program == "plot":
        plot_function()
    if program == "test":
        run_project_test()
