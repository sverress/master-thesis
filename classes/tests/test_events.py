import unittest
from classes import (
    ScooterDeparture,
    ScooterArrival,
    VehicleArrival,
    Event,
    GenerateScooterTrips,
    LostTrip,
    World,
)
from globals import ITERATION_LENGTH_MINUTES


class EventsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(40, initial_location_depot=False)
        self.large_world = World(
            40, sample_size=500, number_of_clusters=20, initial_location_depot=False
        )
        self.world.state.current_location = self.world.state.clusters[0]
        self.large_world.state.current_location = self.large_world.state.clusters[0]
        self.departure_time = 1
        self.travel_time = 5

    def test_scooter_departure(self):
        departure_event = ScooterDeparture(
            self.departure_time, self.world.state.current_location.id
        )

        departure_event.perform(self.world)

        # test if the time of world object is set to the departure time
        self.assertEqual(departure_event.time, self.world.time)

        arrival_event = next(iter(self.world.stack))

        # test if the arrival event created in departure has the same departure cluster id
        self.assertEqual(
            departure_event.departure_cluster_id, arrival_event.departure_cluster_id,
        )

        # scooter should have been removed from the scooters in the state
        self.assertFalse(
            self.world.state.get_scooters().__contains__(arrival_event.scooter)
        )

    def test_scooter_arrival(self):
        self.world.state.current_location = self.world.state.clusters[0]
        scooter = self.world.state.current_location.get_valid_scooters(20.0)[0]

        scooter_battery = scooter.battery

        arrival_cluster = self.world.state.get_random_cluster()

        arrival_event = ScooterArrival(
            self.departure_time + self.travel_time,
            scooter,
            arrival_cluster.id,
            self.world.state.get_random_cluster(exclude=arrival_cluster).id,
            3,
        )

        arrival_event.perform(self.world)

        # test til world time is set to arrival time after arrival is performed
        self.assertEqual(self.world.time, self.departure_time + self.travel_time)

        # test if arrival cluster contains the arrived scooter
        self.assertTrue(arrival_cluster.scooters.__contains__(scooter))

        # test if battery has decreased
        self.assertLess(scooter.battery, scooter_battery)

    def test_vehicle_arrival(self):
        random_cluster_in_state = self.world.state.get_random_cluster(
            exclude=self.world.state.current_location
        )
        # Create a vehicle arrival event with a arrival time of 20 arriving at a random cluster in the world state
        vehicle_arrival = VehicleArrival(20, random_cluster_in_state.id)

        # Perform the vehicle arrival event
        vehicle_arrival.perform(self.large_world)

        # test if the time of world object is set to the departure time
        self.assertEqual(vehicle_arrival.time, self.large_world.time)

        # New current cluster is not the arrival cluster, as the action takes the state to a new cluster
        self.assertNotEqual(
            random_cluster_in_state.id, self.large_world.state.current_location.id
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

        # check if all departure times are inside the interval of iteration duration
        self.assertTrue(
            all(
                [
                    True if 0 <= event.time <= ITERATION_LENGTH_MINUTES else False
                    for event in self.large_world.stack
                ]
            )
        )

    def test_lost_trip(self):
        lost_trip = LostTrip(2)
        lost_trip.perform(self.world)

        # check if lost trip gives negative reward
        self.assertLess(sum(self.world.rewards), 0)

    def test_try_to_move_back_in_time(self):
        # Create a nice event to make time fly
        event = Event(10)
        event.perform(self.world)
        # Create an event that tries to move back in time
        wrong_event = Event(5)
        self.assertRaises(ValueError, wrong_event.perform, self.world)


if __name__ == "__main__":
    unittest.main()
