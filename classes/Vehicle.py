from classes.Scooter import Scooter


class Vehicle:
    def __init__(self):
        self.battery_inventory = 20
        self.scooter_inventory = []
        self.scooter_inventory_capacity = 5

    def change_batteries(self, scooters: Scooter):
        self.battery_inventory -= len(scooters)
        for scooter in scooters:
            scooter.change_battery()

    def pick_up(self, scooter: Scooter):
        if len(self.scooter_inventory) + 1 > self.scooter_inventory_capacity:
            return False
        else:
            self.scooter_inventory.append(scooter)
            self.battery_inventory -= 1
            scooter.change_battery()
            return True

    def drop_off(self, scooter: Scooter):
        if scooter in self.scooter_inventory:
            self.scooter_inventory.remove(scooter)
            return True
        else:
            return False
