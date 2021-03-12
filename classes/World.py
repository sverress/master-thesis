import clustering.scripts as clustering_scripts
import numpy as np
import bisect


class World:
    def __init__(self, shift_duration: int, sample_size=100, number_of_clusters=20):
        self.shift_duration = shift_duration
        self.state = clustering_scripts.get_initial_state(
            sample_size=sample_size, number_of_clusters=number_of_clusters
        )
        self.stack = []
        self.time = 0
        self.rewards = []
        self.cluster_flow = {
            (start, end): 0
            for start in np.arange(len(self.state.clusters))
            for end in np.arange(len(self.state.clusters))
            if start != end
        }

    def run(self):
        while self.time < self.shift_duration:
            self.stack.pop(0).perform(self)

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

    def add_event(self, event) -> None:
        """
        Adds event to the sorted stack.
        Avoids calling sort on every iteration by using the bisect package
        :param event: event to insert
        """
        insert_index = bisect.bisect([event.time for event in self.stack], event.time)
        self.stack.insert(insert_index, event)

    def add_trip_to_flow(self, start: int, end: int) -> None:
        """
        Adds a trip from start to end for cluster flow
        :param start: departure cluster
        :param end: arrival cluster
        """
        self.cluster_flow[(start, end)] += 1

    def get_cluster_flow(self) -> [(int, int, int)]:
        """
        Get all flows between cluster since last vehicle arrival
        :return: list: tuple (start, end, flow) flow from departure cluster to arrival cluster
        """
        return [(start, end, flow) for (start, end), flow in self.cluster_flow.items()]

    def clear_flow_dict(self) -> None:
        """
        Clears the cluster flow dict
        """
        for key in self.cluster_flow.keys():
            self.cluster_flow[key] = 0
