import unittest
import numpy as np
from classes import World
from classes.events import ScooterDeparture, ScooterArrival, VehicleArrival, Event
import random


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

        # scooter should have been removed from the scooters in the state
        self.assertFalse(self.world.state.get_scooters().__contains__(scooter))

    def test_scooter_arrival(self):
        scooter = self.world.state.current_cluster.get_valid_scooters(20.0)[0]

        scooter_battery = scooter.battery

        arrival_cluster = random.choice(self.world.state.clusters)

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

    def test_vehicle_arrival(self):
        random_cluster_in_state = random.choice(
            [
                cluster
                for cluster in self.world.state.clusters
                if cluster.id != self.world.state.current_cluster.id
            ]
        )
        # Create a vehicle arrival event with a arrival time of 20 arriving at a random cluster in the world state
        vehicle_arrival = VehicleArrival(20, random_cluster_in_state.id)

        # Perform the vehicle arrival event
        vehicle_arrival.perform(self.world)

        # The total reward has increased
        self.assertLess(0, self.world.get_total_reward())

        # test if the time of world object is set to the departure time
        self.assertEqual(vehicle_arrival.time, self.world.time)

        # New current cluster is not the arrival cluster, as the action takes the state to a new cluster
        self.assertNotEqual(
            random_cluster_in_state.id, self.world.state.current_cluster.id
        )

        # Vehicle arrival event created a new vehicle arrival event
        self.assertEqual(1, len(self.world.stack))

    def test_try_to_move_back_in_time(self):
        # Create a nice event to make time fly
        event = Event(10)
        event.perform(self.world)
        # Create an event that tries to move back in time
        wrong_event = Event(5)
        self.assertRaises(ValueError, wrong_event.perform, self.world)


if __name__ == "__main__":
    unittest.main()
