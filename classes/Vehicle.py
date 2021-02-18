from classes.Scooter import Scooter


class Vehicle:
    def __init__(self):
        self.battery_inventory = 20
        self.scooter_inventory = []
        self.scooter_inventory_capacity = 5

    def change_battery(self, scooter: Scooter):
        if self.battery_inventory <= 0:
            return False
        else:
            self.battery_inventory -= 1
            scooter.swap_battery()
            return True

    def pick_up(self, scooter: Scooter):
        if len(self.scooter_inventory) + 1 > self.scooter_inventory_capacity:
            return False
        else:
            self.scooter_inventory.append(scooter)
            self.battery_inventory -= 1
            scooter.swap_battery()
            return True

    def drop_off(self, scooter: Scooter):
        if scooter in self.scooter_inventory:
            self.scooter_inventory.remove(scooter)
            return True
        else:
            return False
