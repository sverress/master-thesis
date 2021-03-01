"""
A file for testing stuff without dealing with circular imports
"""
from clustering.scripts import get_initial_state
from system_simulation.scripts import system_simulate
from visualization.visualizer import *
from classes.Cluster import Cluster
from classes.Scooter import Scooter
import unittest


class BasicVisualizerTests(unittest.TestCase):
    @staticmethod
    def test_state_and_flow_between_clusters():
        # state and flow between clusters visualization
        state = get_initial_state(sample_size=50, number_of_clusters=5)

        state.visualize()

        flows, trips = system_simulate(state)

        visualize_cluster_flow(state, flows)

    @staticmethod
    def test_scooter_trips():
        # scooter trips visualization

        current_state = get_initial_state(sample_size=20, number_of_clusters=5)

        next_state = copy.deepcopy(current_state)

        flows, scooter_trips = next_state.system_simulate()

        visualize_scooter_simulation(
            current_state,
            next_state,
            Action(
                [Scooter(0, 0, 69.0, 1)],
                [Scooter(0, 0, 69.0, 2)],
                [Scooter(0, 0, 69.0, 3)],
                Cluster(2, [Scooter(59, 10, 69.0, 1)]),
            ),
            scooter_trips,
        )


if __name__ == "__main__":
    unittest.main()
