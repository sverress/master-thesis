import os
import unittest

import analysis.evaluate_policies
import analysis.train_value_function
import analysis.multiprocessing_training
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
            80,
            None,
            clustering.scripts.get_initial_state(100, 10),
            visualize=False,
            verbose=False,
        )

    def test_run_analysis(self):
        # Runs random and do nothing policies
        analysis.evaluate_policies.run_analysis([], self.world)

    @staticmethod
    def test_run_analysis_from_path():
        analysis.evaluate_policies.run_analysis_from_path(
            "world_cache/trained_models/LinearValueFunction/c30_s2500/TEST_SET",
            runs_per_policy=1,
            shift_duration=80,
        )

    def delete_dir(self, training_directory):
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

    def test_train_value_function(self):
        self.world.policy = self.world.set_policy(self.rollout_value_func_policy)
        analysis.train_value_function.train_value_function(
            self.world, training_shifts_before_save=1, models_to_be_saved=2
        )
        # Remove created files
        training_directory = os.path.join(
            globals.WORLD_CACHE_DIR, self.world.get_train_directory()
        )
        self.delete_dir(training_directory)

    def test_train_multiprocessing(self):
        shifts = [1, 2, 3]
        analysis.multiprocessing_training.multiprocess_train(
            shifts,
            analysis.multiprocessing_training.run_train_with_shift_duration,
        )
        # Fake world for printing and file directoryies
        world = classes.World(
            globals.SHIFT_DURATION,
            None,
            clustering.scripts.get_initial_state(2500, 30),
            verbose=False,
            visualize=False,
        )
        world.policy = world.set_policy(
            decision.EpsilonGreedyValueFunctionPolicy(
                decision.value_functions.ANNValueFunction([10])
            )
        )
        # Remove created files
        for directory in [
            os.path.join(
                globals.WORLD_CACHE_DIR,
                world.get_train_directory(f"shift_{shift}"),
            )
            for shift in shifts
        ]:
            self.delete_dir(directory)
