"""
A file for testing stuff without dealing with circular imports
"""
from clustering.scripts import get_initial_state
from system_simulation.scripts import system_simulate
from visualization.visualizer import *
from analysis.evaluate_policies import run_analysis
import unittest
import decision


class BasicVisualizerTests(unittest.TestCase):
    @staticmethod
    def test_state_and_flow_between_clusters():
        # state and flow between clusters visualization
        state = get_initial_state(sample_size=100, number_of_clusters=6)

        state.visualize()

        flows, trips, _ = system_simulate(state)

        visualize_cluster_flow(state, flows)

    @staticmethod
    def test_scooter_trips():
        # scooter trips visualization

        current_state = get_initial_state(sample_size=20, number_of_clusters=5)

        next_state = copy.deepcopy(current_state)

        flows, scooter_trips, _ = next_state.system_simulate()

        current_state.visualize_system_simulation(scooter_trips)

        current_state.visualize_flow(flows)

    @staticmethod
    def test_visualize_clusters():
        current_state = get_initial_state(sample_size=1000, number_of_clusters=20)
        current_state.visualize_clustering()

    @staticmethod
    def test_analysis():
        policy = decision.SwapAllPolicy()
        # test the analysis plot
        run_analysis(
            [policy],
            classes.World(
                60, None, get_initial_state(sample_size=100, number_of_clusters=10)
            ),
            smooth_curve=False,
            runs_per_policy=1,
        )


if __name__ == "__main__":
    unittest.main()
