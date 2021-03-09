from classes.events import Event, LostTrip, ScooterArrival, ScooterDeparture
from globals import BATTERY_LIMIT, SCOOTER_SPEED
import numpy as np


class GenerateScooterTrips(Event):
    def __init__(self, time: int):
        super().__init__(time)

    def perform(self, world) -> None:
        cluster_indices = np.arange(len(world.state.clusters))

        for departure_cluster in world.state.clusters:
            # poisson process to select number of trips in a iteration
            number_of_trips = round(
                np.random.poisson(departure_cluster.trip_intensity_per_iteration)
            )

            # generate trip departure times (can be implemented with np.random.uniform if we want decimal times)
            # both functions generate numbers from a discrete uniform distribution
            trips_departure_time = sorted(np.random.randint(0, 20, number_of_trips))

            # get all available scooter in the cluster
            available_scooters = departure_cluster.get_valid_scooters(BATTERY_LIMIT)

            # generate departure and arrival event for every trip and add to world stack
            for departure_time in trips_departure_time:
                # if there are no more available scooters -> make a LostTrip event for that departure time
                if len(available_scooters) > 0:
                    scooter = available_scooters.pop()
                else:
                    world.add_event(LostTrip(departure_time))
                    continue

                # add departure event to the stack
                world.add_event(
                    ScooterDeparture(departure_time, departure_cluster.id, scooter)
                )

                arrival_cluster_index = np.random.choice(
                    cluster_indices, p=departure_cluster.get_leave_distribution()
                )

                trip_distance = world.state.get_distance(
                    departure_cluster.id, arrival_cluster_index
                )

                # calculate arrival time
                arrival_time = departure_time + round(trip_distance / SCOOTER_SPEED)

                world.add_event(
                    ScooterArrival(
                        arrival_time, scooter, arrival_cluster_index, trip_distance
                    )
                )

        super(GenerateScooterTrips, self).perform(world)
