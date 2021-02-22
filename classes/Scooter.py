class Scooter:
    def __init__(self, lat: float, lon: float, battery: float, id: int):
        self.lat = lat
        self.lon = lon
        self.battery = battery
        self.id = id
        self.battery_change_per_kilometer = 5.0

    def travel(self, distance):
        self.battery -= distance * self.battery_change_per_kilometer

    def swap_battery(self):
        self.battery = 100.0

    def __repr__(self):
        return f"{self.id}: battery: {self.battery}"
