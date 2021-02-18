"""
A file for testing stuff without dealing with circular imports
"""
from clustering.scripts import get_initial_state
from system_simulation.scripts import system_simulate
from visualization.visualizer import visualize_state

if __name__ == "__main__":
    state = get_initial_state()
    print("------------ Before system simulation ------------ \n")
    visualize_state(state)
    state.visualize_clustering()
    for i, cluster in enumerate(state.clusters):
        print(cluster.__str__())

    system_simulate(state)

    print("------------ After system simulation ------------ \n")
    visualize_state(state)
    state.visualize_clustering()
    for cluster in state.clusters:
        print(cluster.__str__())
