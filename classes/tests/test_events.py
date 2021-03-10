import unittest
import numpy as np
from classes import World
from classes import (
    ScooterDeparture,
    ScooterArrival,
    VehicleArrival,
    Event,
    GenerateScooterTrips,
    LostTrip,
)

import random


class EventsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(40)
        self.large_world = World(40, sample_size=2000, number_of_clusters=20)
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

    def test_vehicle_arrival(self):
        random_cluster_in_state = random.choice(
            [
                cluster
                for cluster in self.large_world.state.clusters
                if cluster.id != self.large_world.state.current_cluster.id
            ]
        )
        # Create a vehicle arrival event with a arrival time of 20 arriving at a random cluster in the world state
        vehicle_arrival = VehicleArrival(20, random_cluster_in_state.id)

        # Perform the vehicle arrival event
        vehicle_arrival.perform(self.large_world)

        # The total reward has increased
        self.assertLess(0, self.large_world.get_total_reward())

        # test if the time of world object is set to the departure time
        self.assertEqual(vehicle_arrival.time, self.large_world.time)

        # New current cluster is not the arrival cluster, as the action takes the state to a new cluster
        self.assertNotEqual(
            random_cluster_in_state.id, self.large_world.state.current_cluster.id
        )

        # Vehicle arrival event created a new vehicle arrival event
        self.assertEqual(1, len(self.large_world.stack))

    def test_generate_scooter_trips(self):
        generate_trips_event = GenerateScooterTrips(0)

        generate_trips_event.perform(self.large_world)

        # check if any trips or GenerateScooterTrip object is created
        self.assertGreater(len(self.large_world.stack), 0)

        # check if world stack contains a GenerateScooterTrips object
        self.assertTrue(
            any(
                isinstance(event, GenerateScooterTrips)
                for event in self.large_world.stack
            )
        )

        departures = [
            event
            for event in self.large_world.stack
            if isinstance(event, ScooterDeparture)
        ]
        arrivals = [
            event
            for event in self.large_world.stack
            if isinstance(event, ScooterArrival)
        ]

        # check if there is equally many departures and arrivals
        self.assertEqual(len(departures), len(arrivals))

        trips = [
            (departure, arrival)
            for departure in departures
            for arrival in arrivals
            if departure.scooter.id == arrival.scooter.id
        ]

        # check if departure time is less then arrival time
        for trip in trips:
            departure_event = trip[0]
            arrival_event = trip[1]
            self.assertLess(departure_event.time, arrival_event.time)

    def test_try_to_move_back_in_time(self):
        # Create a nice event to make time fly
        event = Event(10)
        event.perform(self.world)
        # Create an event that tries to move back in time
        wrong_event = Event(5)
        self.assertRaises(ValueError, wrong_event.perform, self.world)


if __name__ == "__main__":
    unittest.main()
