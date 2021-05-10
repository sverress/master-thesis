from classes.Location import Location
from globals import *


class Depot(Location):
    def __init__(self, lat: float, lon: float, depot_id: int, main_depot=False):
        super(Depot, self).__init__(lat, lon, depot_id)
        self.capacity = MAIN_DEPOT_CAPACITY if main_depot else SMALL_DEPOT_CAPACITY
        self.time = 0
        self.charging = []

    def swap_battery_inventory(self, time, number_of_battery_to_change) -> int:
        self.capacity += self.get_delta_capacity(time)
        self.time = time

        if number_of_battery_to_change > self.capacity:
            raise ValueError(
                f"Depot has only {self.capacity} batteries available. "
                f"Vehicle tried to swap {number_of_battery_to_change}"
            )

        self.capacity -= number_of_battery_to_change

        self.charging.append((time, number_of_battery_to_change))

        return (
            round(number_of_battery_to_change * SWAP_TIME_PER_BATTERY)
            + CONSTANT_DEPOT_DURATION
        )

    def get_available_battery_swaps(self, time):
        return self.capacity + self.get_delta_capacity(time, pop=False)

    def get_delta_capacity(self, time, pop=True):
        delta_capacity = 0
        for i, (charging_start_time, number_of_batteries) in enumerate(self.charging):
            if time > charging_start_time + CHARGE_TIME_PER_BATTERY:
                delta_capacity += number_of_batteries
                if pop:
                    self.charging.pop(i)
        return delta_capacity

    def __str__(self):
        return f"Depot {self.id}"

    def __repr__(self):
        return f"<Depot, id: {self.id}, cap: {self.capacity}>"
