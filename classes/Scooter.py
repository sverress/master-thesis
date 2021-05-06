from classes.Location import Location


class Scooter(Location):
    def __init__(self, lat: float, lon: float, battery: float, scooter_id: int):
        super().__init__(lat, lon, scooter_id)
        self.battery = battery
        self.battery_change_per_kilometer = 5.0

    def __deepcopy__(self, *args):
        return Scooter(self.lat, self.lon, self.battery, self.id)

    def travel(self, distance):
        self.battery -= distance * self.battery_change_per_kilometer

    def swap_battery(self):
        self.battery = 100.0

    def set_coordinates(self, lat: float, lon: float):
        self.set_location(lat, lon)

    def __repr__(self):
        return f"ID-{self.id} B-{round(self.battery,1)}"
