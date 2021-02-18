"""
A file for testing stuff without dealing with circular imports
"""
from clustering.scripts import get_initial_state
from system_simulation.scripts import system_simulate
from visualization.visualizer import *

if __name__ == "__main__":
    state = get_initial_state(number_of_clusters=6)
    print("------------ Before system simulation ------------ \n")
    visualize_state(state)
    state.visualize_clustering()
    for i, cluster in enumerate(state.clusters):
        print(cluster.__str__())

    trips = system_simulate(state)

    print("------------ After system simulation ------------ \n")
    visualize_simulation(state, trips)
    state.visualize_clustering()
    for cluster in state.clusters:
        print(cluster.__str__())
