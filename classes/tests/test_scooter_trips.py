import unittest
import numpy as np
from classes import World
from classes.events import ScooterDeparture, ScooterArrival


class EventsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(40)
        self.departure_time = 1
        self.travel_time = 5

    def test_scooter_departure(self):
        scooter = self.world.state.current_cluster.get_valid_scooters(20.0)[0]

        departure_event = ScooterDeparture(
            self.departure_time, self.world.state.current_cluster.id, scooter
        )

        departure_event.perform(self.world)

        # test if the time of world object is set to the departure time
        self.assertEqual(departure_event.time, self.world.time)

        scooters = self.world.state.get_scooters()

        # scooter should have been removed from the scooters in the state
        self.assertFalse(scooters.__contains__(scooter))

    def test_scooter_arrival(self):
        scooter = self.world.state.current_cluster.get_valid_scooters(20.0)[0]

        scooter_battery = scooter.battery

        arrival_cluster = self.world.state.get_cluster_by_id(
            round(np.random.uniform(len(self.world.state.clusters)))
        )

        arrival_event = ScooterArrival(
            self.departure_time + self.travel_time, scooter, arrival_cluster.id, 3
        )

        arrival_event.perform(self.world)

        # test til world time is set to arrival time after arrival is performed
        self.assertEqual(self.world.time, self.departure_time + self.travel_time)

        # test if arrival cluster contains the arrived scooter
        self.assertTrue(arrival_cluster.scooters.__contains__(scooter))

        # test if battery has decreased
        self.assertLess(scooter.battery, scooter_battery)


if __name__ == "__main__":
    unittest.main()
