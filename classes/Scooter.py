class Scooter:
    def __init__(self, lat: float, lon: float, battery: float, id: int):
        self.lat = lat
        self.lon = lon
        self.battery = battery
        self.id = id
        self.battery_change_per_kilometer = 1.0

    def change_battery(self, distance):
        self.battery -= distance * self.battery_change_per_kilometer
