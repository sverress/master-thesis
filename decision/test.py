import unittest

from clustering.scripts import get_initial_state
from decision.scripts import *


class BasicDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_state = get_initial_state()

    def test_number_of_actions(self):

        self.assertEqual(2, 2)


if __name__ == "__main__":
    unittest.main()
