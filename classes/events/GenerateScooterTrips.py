from classes.events import Event
from globals import BATTERY_LIMIT
from collections import defaultdict
import numpy as np


class GenerateScooterTrips(Event):
    def __init__(self, time: int):
        super().__init__(time)

    def perform(self, world) -> None:
        trips = defaultdict(list)

        for departure_cluster in world.state.clusters:
            # poisson process to select number of trips in a iteration
            number_of_trips = round(
                np.random.poisson(departure_cluster.trip_intensity_per_iteration)
            )

            # can't complete more trips then there is scooters with battery over min_battery
            valid_scooters = departure_cluster.get_valid_scooters(BATTERY_LIMIT)
            if number_of_trips > len(valid_scooters):
                number_of_trips = len(valid_scooters)

        new_trips = [
            trip for same_departure in trips.values() for trip in same_departure
        ]

        world.add_events(new_trips)
