import unittest
from clustering.scripts import get_initial_state


class BasicSystemSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = get_initial_state(500)

    def test_equal_number_of_scooters(self):
        number_of_scooters_before = sum(
            map(lambda cluster: len(cluster.scooters), self.state.clusters)
        )

        self.state.system_simulate()
        number_of_scooters_after = sum(
            map(lambda cluster: len(cluster.scooters), self.state.clusters)
        )
        self.assertEqual(number_of_scooters_before, number_of_scooters_after)

    def test_less_or_equal_total_battery(self):
        total_battery_of_scooters_before = sum(
            map(
                lambda cluster: sum([scooter.battery for scooter in cluster.scooters]),
                self.state.clusters,
            )
        )

        self.state.system_simulate()
        total_battery_of_scooters_after = sum(
            map(
                lambda cluster: sum([scooter.battery for scooter in cluster.scooters]),
                self.state.clusters,
            )
        )
        self.assertLessEqual(
            total_battery_of_scooters_after, total_battery_of_scooters_before
        )

    def test_flows_and_trips(self):
        initial_cluster_inventory = {}

        for cluster in self.state.clusters:
            initial_cluster_inventory[cluster.id] = len(cluster.scooters)

        flows, trips, _ = self.state.system_simulate()

        out_flow = {cluster.id: 0 for cluster in self.state.clusters}
        in_flow = {cluster.id: 0 for cluster in self.state.clusters}

        trips_from = {cluster.id: 0 for cluster in self.state.clusters}
        trips_to = {cluster.id: 0 for cluster in self.state.clusters}

        for start, end, scooter in trips:
            trips_from[start.id] += 1
            trips_to[end.id] += 1

        for cluster in self.state.clusters:
            out_flow[cluster.id] = sum(
                [flow for start, end, flow in flows if start == cluster.id]
            )
            in_flow[cluster.id] = sum(
                [flow for start, end, flow in flows if end == cluster.id]
            )

            # Test flow in/out equal to change in scooter inventory in a cluster
            self.assertEqual(
                initial_cluster_inventory[cluster.id] - len(cluster.scooters),
                out_flow[cluster.id] - in_flow[cluster.id],
            )

            # Test number of trips in/out equal to change in scooter inventory in a cluster
            self.assertEqual(
                initial_cluster_inventory[cluster.id] - len(cluster.scooters),
                trips_from[cluster.id] - trips_to[cluster.id],
            )

        # Test that total flow out equal to total flow in
        self.assertEqual(sum(out_flow.values()), sum(in_flow.values()))

    def test_lost_demand(self):
        for cluster in self.state.clusters:
            cluster.trip_intensity_per_iteration = 5
            for scooter in cluster.scooters:
                scooter.battery = 0

        _, _, lost_demand = self.state.system_simulate()

        self.assertGreater(lost_demand, 0)


if __name__ == "__main__":
    unittest.main()
