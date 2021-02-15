from classes.Scooter import Scooter


class Vehicle:
    def __init__(self):
        self.battery_inventory = 20
        self.scooter_inventory = []
        self.scooter_inventory_capacity = 5

    def change_batteries(self, amount: int):
        self.battery_inventory -= amount

    def pick_up(self, scooter: Scooter):
        if len(self.scooter_inventory) + 1 > self.scooter_inventory_capacity:
            return False
        else:
            self.scooter_inventory.append(scooter)
            return True
