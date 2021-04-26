import unittest

import analysis.evaluate_policies
import classes
import clustering.scripts
import decision
import decision.value_functions


class EvaluatePoliciesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rollout_value_func_policy = decision.RolloutValueFunctionPolicy(
            decision.EpsilonGreedyValueFunctionPolicy(
                decision.value_functions.LinearValueFunction()
            )
        )
        self.random_rollout_policy = decision.RandomRolloutPolicy()
        self.world = classes.World(
            5,
            None,
            clustering.scripts.get_initial_state(100, 10),
            visualize=False,
            verbose=False,
        )

    def test_run_analysis(self):
        analysis.evaluate_policies.run_analysis(
            [self.random_rollout_policy, self.rollout_value_func_policy], self.world
        )
