import unittest
from clustering.scripts import get_initial_state
from classes.World import World
from classes.events.VehicleArrival import VehicleArrival
from globals import BATTERY_INVENTORY, SWAP_TIME_PER_BATTERY


class DepotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(
            shift_duration=BATTERY_INVENTORY * SWAP_TIME_PER_BATTERY + 1,
            sample_size=100,
            number_of_clusters=10,
            initial_state=get_initial_state(500),
        )
        self.world.state.current_location = self.world.state.depots[0]

    def test_depot_charge(self):
        self.world.stack.append(
            VehicleArrival(0, self.world.state.current_location.id, visualize=False)
        )
        self.world.state.vehicle.battery_inventory = 0
        self.world.run()
        self.assertGreater(self.world.state.vehicle.battery_inventory, 0)

    def test_vehicle_battery_inventory_change(self):
        pass


if __name__ == "__main__":
    unittest.main()
