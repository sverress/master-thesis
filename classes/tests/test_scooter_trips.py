import unittest
from classes import World
from classes.events import ScooterDeparture, VehicleArrival


class ScooterTripsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(40)

    def test_scooter_departure(self):
        scooter = self.world.state.current_cluster.get_valid_scooters(20.0)

        departure_event = ScooterDeparture(2)

        vehicle = VehicleArrival(2)

        self.world.stack = []


if __name__ == "__main__":
    unittest.main()
