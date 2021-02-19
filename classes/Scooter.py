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

    def change_coordinates(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return f"ID: {self.id} B: {self.battery}"
