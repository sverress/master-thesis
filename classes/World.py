import clustering.scripts as clustering_scripts
from classes.events import Event
import bisect

from decision.policies import RandomRolloutPolicy


class World:
    def __init__(
        self,
        shift_duration: int,
        sample_size=100,
        number_of_clusters=20,
        policy=RandomRolloutPolicy,
    ):
        self.shift_duration = shift_duration
        self.state = clustering_scripts.get_initial_state(
            sample_size=sample_size, number_of_clusters=number_of_clusters
        )
        self.stack = []
        self.time = 0
        self.rewards = []
        self.policy = policy

    def run(self):
        while self.time < self.shift_duration:
            self.stack.pop().perform(self)

    def get_remaining_time(self) -> int:
        """
        Computes the remaining time by taking the difference between the shift duration
        and the current time of the world object.
        :return: the remaining time as a float
        """
        return self.shift_duration - self.time

    def add_reward(self, reward: float) -> None:
        """
        Adds the input reward to the rewards list of the world object
        :param reward: reward given
        """
        self.rewards.append(reward)

    def get_total_reward(self) -> float:
        """
        Get total accumulated reward at current point of time
        :return:
        """
        return sum(self.rewards)

    def add_event(self, event: Event) -> None:
        """
        Adds event to the sorted stack.
        Avoids calling sort on every iteration by using the bisect package
        :param event: event to insert
        """
        insert_index = bisect.bisect([event.time for event in self.stack], event.time)
        self.stack.insert(insert_index, event)
