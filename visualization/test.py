"""
A file for testing stuff without dealing with circular imports
"""
from clustering.scripts import get_initial_state
from system_simulation.scripts import system_simulate
from visualization.visualizer import *

if __name__ == "__main__":
    state = get_initial_state(sample_size=100, number_of_clusters=10)
    print("\n ------------ Before system simulation ------------ \n")
    visualize_state(state)
    for i, cluster in enumerate(state.clusters):
        print(cluster)

    trips = system_simulate(state)

    print("\n ------------ After system simulation ------------ \n")
    visualize_cluster_flow(state, trips)
    for cluster in state.clusters:
        print(cluster)
