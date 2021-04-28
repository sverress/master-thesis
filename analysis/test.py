import os
import unittest

import analysis.evaluate_policies
import analysis.train_value_function
import classes
import clustering.scripts
import decision
import decision.value_functions
import globals


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rollout_value_func_policy = decision.EpsilonGreedyValueFunctionPolicy(
            decision.value_functions.LinearValueFunction()
        )
        self.random_rollout_policy = decision.RandomRolloutPolicy(number_of_rollouts=2)
        self.world = classes.World(
            2,
            None,
            clustering.scripts.get_initial_state(100, 10),
            visualize=False,
            verbose=False,
        )

    def test_run_analysis(self):
        analysis.evaluate_policies.run_analysis(
            [self.random_rollout_policy, self.rollout_value_func_policy], self.world
        )

    @staticmethod
    def test_run_analysis_from_path():
        analysis.evaluate_policies.run_analysis_from_path(
            "world_cache/trained_models/LinearValueFunction/c10_s100/TEST_EVALUATE_FROM_PATH_DO_NOT_DELETE"
        )

    def test_train_value_function(self):
        self.world.policy = self.world.set_policy(self.rollout_value_func_policy)
        analysis.train_value_function.train_value_function(
            self.world, training_shifts_before_save=1, models_to_be_saved=2
        )
        # Removed created files
        training_directory = os.path.join(
            globals.WORLD_CACHE_DIR, self.world.get_train_directory()
        )
        for world_file_path in os.listdir(training_directory):
            file_path = os.path.join(training_directory, world_file_path)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                self.assertTrue(
                    False,
                    "file not found. Possible that train value function did not create necessary files",
                )

        if os.path.isdir(training_directory):
            os.rmdir(training_directory)
        else:
            self.assertTrue(
                False,
                "file not found. Possible that train value function did not create necessary files",
            )
