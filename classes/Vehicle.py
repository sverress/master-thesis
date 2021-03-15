from classes.Scooter import Scooter
from globals import BATTERY_INVENTORY, SCOOTER_INVENTORY


class Vehicle:
    def __init__(self):
        self.battery_inventory = BATTERY_INVENTORY
        self.scooter_inventory = []
        self.scooter_inventory_capacity = SCOOTER_INVENTORY

    def change_battery(self, scooter: Scooter):
        if self.battery_inventory <= 0:
            raise ValueError(
                "Can't change battery when the vehicle's battery inventory is empty"
            )
        else:
            self.battery_inventory -= 1
            scooter.swap_battery()
            return True

    def pick_up(self, scooter: Scooter):
        if len(self.scooter_inventory) + 1 > self.scooter_inventory_capacity:
            raise ValueError("Can't pick up an scooter when the vehicle is full")
        else:
            self.scooter_inventory.append(scooter)
            self.change_battery(scooter)
            scooter.remove_location()

    def drop_off(self, scooter_id: int):
        if scooter_id not in map(lambda scooter: scooter.id, self.scooter_inventory):
            raise ValueError(
                "Can't deliver a scooter that isn't in the vehicle inventory"
            )

        scooter = next(
            (scooter for scooter in self.scooter_inventory if scooter.id == scooter_id),
            None,
        )
        self.scooter_inventory.remove(scooter)
        return scooter
