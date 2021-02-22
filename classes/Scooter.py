from classes import Location


class Scooter:
    def __init__(self, lat: float, lon: float, battery: float, id: int):
        self.location = Location(lat, lon)
        self.battery = battery
        self.id = id
        self.battery_change_per_kilometer = 5.0

    def travel(self, distance):
        self.battery -= distance * self.battery_change_per_kilometer

    def swap_battery(self):
        self.battery = 100.0

    def change_coordinates(self, lat: float, lon: float):
        self.location.set_location(lat, lon)

    def __repr__(self):
        return f"ID: {self.id} B: {self.battery}"
