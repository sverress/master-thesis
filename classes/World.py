import clustering.scripts as clustering_scripts
import numpy as np
import bisect
import classes
from globals import BATTERY_LIMIT, LOST_TRIP_REWARD, ITERATION_LENGTH_MINUTES, WHITE
from decision.get_policy import get_policy
from progress.bar import IncrementalBar


class World:
    class WorldMetric:
        def __init__(self):
            self.lost_demand = []
            self.average_deviation_ideal_state = []
            self.deficient_battery = []
            self.time = []

        def add_analysis_metrics(
            self, rewards: [int], clusters: [classes.Cluster], time: int
        ):
            self.lost_demand.append(
                sum([1 for reward in rewards if reward == LOST_TRIP_REWARD])
                if len(rewards) > 0
                else 0
            )
            self.average_deviation_ideal_state.append(
                sum(
                    [
                        abs(
                            (
                                sum(
                                    [
                                        1
                                        for _ in cluster.get_valid_scooters(
                                            BATTERY_LIMIT
                                        )
                                    ]
                                )
                            )
                            - cluster.ideal_state
                        )
                        for cluster in clusters
                    ]
                )
                / len(clusters)
            )
            self.deficient_battery.append(
                sum(
                    [
                        cluster.ideal_state * 100
                        - (
                            sum(
                                [
                                    scooter.battery
                                    for scooter in cluster.get_valid_scooters(
                                        BATTERY_LIMIT
                                    )
                                ]
                            )
                        )
                        for cluster in clusters
                        if len(cluster.scooters) < cluster.ideal_state
                    ]
                )
            )
            self.time.append(time)

        def get_lost_demand(self):
            return self.lost_demand

        def get_deviation_ideal_state(self):
            return self.average_deviation_ideal_state

        def get_deficient_battery(self):
            return self.deficient_battery

        def get_time_array(self):
            return self.time

        def get_all_metrics(self):
            return (
                self.lost_demand,
                self.average_deviation_ideal_state,
                self.deficient_battery,
            )

    def __init__(
        self,
        shift_duration: int,
        sample_size=100,
        number_of_clusters=20,
        policy="RandomRolloutPolicy",
    ):
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
        self.policy = get_policy(policy)
        self.metrics = World.WorldMetric()
        self.progress_bar = IncrementalBar(
            "Running World",
            check_tty=False,
            max=round(shift_duration / ITERATION_LENGTH_MINUTES) + 1,
            color=WHITE,
            suffix="%(percent)d%% - ETA %(eta)ds",
        )

    def run(self):
        while self.time < self.shift_duration:
            event = self.stack.pop(0)
            event.perform(self)
            event.add_metric(self, self.time)
            if isinstance(event, classes.GenerateScooterTrips):
                self.progress_bar.next()
        self.progress_bar.finish()

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

    def get_scooters_on_trip(self) -> [(int, int, classes.Scooter)]:
        """
        Get all scooters that are currently out on a trip
        :return: list of all scooters that are out on a trip
        """
        return [
            (event.departure_cluster_id, event.arrival_cluster_id, event.scooter)
            for event in self.stack
            if isinstance(event, classes.ScooterArrival)
        ]
