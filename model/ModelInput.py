import pandas as pd
from math import sqrt, pi, sin, cos, atan2
from itertools import product


class ModelInput:
    """
    Sets, Constants, and Parameters for a model instance
    """

    def __init__(
        self,
        scooter_list: pd.DataFrame,
        delivery_nodes_list: pd.DataFrame,
        depot_location: tuple,
        service_vehicles_dict: dict,
    ):
        """
        Creating all input to the gurobi model
        :param scooter_list: list of list - [[lat,lon,battery]*n]
        :param delivery_nodes_list: list of list - [[lat,lon]*m]
        :param depot_location: tuple - (lat,lon)
        :param service_vehicles_dict: dict - ["type"]: (#numbers, scooter capacity, battery capacity)
        """

        # Sets
        self.locations = range(
            1 + len(scooter_list.index) + len(delivery_nodes_list.index)
        )
        self.scooters = self.locations[1 : len(scooter_list.index) + 1]
        self.delivery = self.locations[len(scooter_list.index) + 1 :]
        self.service_vehicles = range(sum(x[0] for x in service_vehicles_dict.values()))
        self.depot = 0

        # Constants
        self.num_scooters = len(scooter_list)
        self.num_locations = len(self.locations)
        self.num_service_vehicles = len(self.service_vehicles)

        # Parameters
        self.reward = (
            [0.0]
            + [1 - x / 100 for x in scooter_list["battery"]]
            + [1.0] * len(delivery_nodes_list.index)
        )  # Reward for visiting location i (uniform distribution). Eks for 3 locations [0, 0.33, 0.66]
        self.time_cost = self.compute_time_matrix(
            scooter_list, delivery_nodes_list, depot_location
        )  # Calculate distance in time between all locations
        self.T_max = 60 * 0.8  # Duration of shift in minutes
        self.Q_b = []
        self.Q_s = []
        for vehicle_type in service_vehicles_dict.keys():
            num_vehicles, scooter_cap, battery_cap = service_vehicles_dict[vehicle_type]
            for i in range(num_vehicles):
                self.Q_b.append(battery_cap)
                self.Q_s.append(scooter_cap)

    def compute_time_matrix(self, scooters, delivery_nodes, depot):
        locations = (
            [depot]
            + list(zip(scooters["lat"], scooters["lon"]))
            + list(zip(delivery_nodes["lat"], delivery_nodes["lon"]))
        )

        return {
            (i, j): ModelInput.compute_distance(
                locations[i][0], locations[i][1], locations[j][0], locations[j][1]
            )
            / (20 * (1000 / 60))
            for i, j in list(product(self.locations, repeat=2))
        }

    @staticmethod
    def compute_distance(lat1, lon1, lat2, lon2):
        """
        Compute the distance between two points in meters
        :param lat1: Coordinate 1 lat
        :param lon1: Coordinate 1 lon
        :param lat2: Coordinate 2 lat
        :param lon2: Coordinate 2 lon
        :return: Meters between coordinates
        """
        radius = 6378.137
        d_lat = lat2 * pi / 180 - lat1 * pi / 180
        d_lon = lon2 * pi / 180 - lon1 * pi / 180
        a = sin(d_lat / 2) * sin(d_lat / 2) + cos(lat1 * pi / 180) * cos(
            lat2 * pi / 180
        ) * sin(d_lon / 2) * sin(d_lon / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        d = radius * c
        return d * 1000
