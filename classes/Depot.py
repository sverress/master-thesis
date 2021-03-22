import Location
from globals import (
    MAIN_DEPOT_CAPACITY,
    SMALL_DEPOT_CAPACITY,
    SWAP_TIME_PER_BATTERY_SMALL_DEPOT,
)


class Depot(Location):
    def __init__(self, lat: float, lon: float, depot_id: int, main_depot=True):
        super(Depot, self).__init__(lat, lon)
        self.main_depot = main_depot
        self.capacity = MAIN_DEPOT_CAPACITY if self.main_depot else SMALL_DEPOT_CAPACITY
        self.time = 0
        self.id = depot_id

    def swap_battery_inventory(self, time, number_of_battery_to_change):
        if not self.main_depot:
            self.charge_batteries(time)

        if number_of_battery_to_change > self.capacity:
            raise ValueError(
                f"Depot has only {self.capacity} batteries available. "
                f"Vehicle tried to swap {number_of_battery_to_change}"
            )

        self.capacity -= number_of_battery_to_change

    def charge_batteries(self, time):
        delta_capacity = round(time - self.time) / SWAP_TIME_PER_BATTERY_SMALL_DEPOT
        self.capacity = (
            self.capacity + delta_capacity
            if self.capacity + delta_capacity < SMALL_DEPOT_CAPACITY
            else SMALL_DEPOT_CAPACITY
        )
        self.time = time
