import pandas as pd
import tsp
from math import sqrt, pi, sin, cos, atan2
from itertools import product
from abc import ABC, abstractmethod


class BaseModelInput(ABC):
    """
    Sets, Constants, and Parameters for a model instance
    """

    def __init__(
        self,
        scooter_list: pd.DataFrame,
        delivery_nodes_list: pd.DataFrame,
        depot_location: tuple,
        service_vehicles_dict: dict,
        optimal_state: list,
        T_max: int,
    ):
        """
        Creating all input to the gurobi model
        :param scooter_list: list of list - [[lat,lon,battery]*n]
        :param delivery_nodes_list: list of list - [[lat,lon]*m]
        :param depot_location: tuple - (lat,lon)
        :param service_vehicles_dict: dict - ["type"]: (#numbers, scooter capacity, battery capacity)
        :param optimal_state: list of optimal state for each zone of the problem
        :param T_max: time limit for vehicles
        """

        # Sets
        self.locations = range(
            1 + len(scooter_list.index) + len(delivery_nodes_list.index)
        )
        self.scooters = self.locations[1 : len(scooter_list.index) + 1]
        self.delivery = self.locations[len(scooter_list.index) + 1 :]
        self.service_vehicles = range(sum(x[0] for x in service_vehicles_dict.values()))
        self.depot = 0
        self.zones = sorted(scooter_list.zone.unique())
        self.zone_scooters = [
            list(
                scooter_list.loc[scooter_list["zone"] == i].index.union(
                    delivery_nodes_list.loc[delivery_nodes_list["zone"] == i].index
                )
            )
            for i in self.zones
        ]
        self.demand_zones = [
            z
            for z in self.zones
            if not all([i in self.scooters for i in self.zone_scooters[z]])
        ]  # Zones with delivery locations

        self.supply_zones = [z for z in self.zones if z not in self.demand_zones]

        # Constants
        self.num_scooters = len(scooter_list)
        self.num_locations = len(self.locations)
        self.num_service_vehicles = len(self.service_vehicles)

        # Parameters
        self.reward = self.compute_reward_matrix(scooter_list, delivery_nodes_list)
        self.time_cost = self.compute_time_matrix(
            scooter_list, delivery_nodes_list, depot_location
        )  # Calculate distance in time between all locations
        if T_max <= 1:
            self.shift_duration = T_max * self.calculate_tsp(
                len(self.locations), self.time_cost
            )
        else:
            self.shift_duration = T_max  # Duration of shift in minutes
        self.battery_level = [0.0] + [
            x / 100 for x in scooter_list["battery"]
        ]  # Battery level of scooter at location i
        self.battery_capacity = []
        self.scooter_capacity = []
        for vehicle_type in service_vehicles_dict.keys():
            num_vehicles, scooter_cap, battery_cap = service_vehicles_dict[vehicle_type]
            for i in range(num_vehicles):
                self.battery_capacity.append(battery_cap)
                self.scooter_capacity.append(scooter_cap)
        self.deviation_from_optimal_state = [
            len(self.zone_scooters[z]) - the_optimal_state
            for z, the_optimal_state in enumerate(optimal_state)
        ]

        # Battery level of scooter at location i
        self.B = [0.0] + [x / 100 for x in scooter_list["battery"]]

    @abstractmethod
    def compute_reward_matrix(self, scooter_list, delivery_nodes_list):
        pass

    def compute_time_matrix(self, scooters, delivery_nodes, depot):
        """
        Computes the time matrix for edges in the network. Time in minutes, calculated by
        euclidean distance and avg vehicle speed of 20 km/h
        :param scooters: pandas DataFrame of all scooters
        :param delivery_nodes: pandas DataFrame of all delivery nodes
        :param depot: tuple
        :return: time matrix all-all
        """
        locations = (
            [depot]
            + list(zip(scooters["lat"], scooters["lon"]))
            + list(zip(delivery_nodes["lat"], delivery_nodes["lon"]))
        )

        return {
            (i, j): BaseModelInput.compute_distance(
                locations[i][0], locations[i][1], locations[j][0], locations[j][1]
            )
            for i, j in list(product(self.locations, repeat=2))
        }

    @staticmethod
    def compute_distance(lat1, lon1, lat2, lon2, speed=20):
        """
        Compute the distance between two points in meters
        :param lat1: Coordinate 1 lat
        :param lon1: Coordinate 1 lon
        :param lat2: Coordinate 2 lat
        :param lon2: Coordinate 2 lon
        :param speed: speed of service vehicle in kilometers per hour
        :return: Meters between coordinates
        """
        minutes_in_an_hour = 60

        radius = 6378.137
        d_lat = lat2 * pi / 180 - lat1 * pi / 180
        d_lon = lon2 * pi / 180 - lon1 * pi / 180
        a = sin(d_lat / 2) * sin(d_lat / 2) + cos(lat1 * pi / 180) * cos(
            lat2 * pi / 180
        ) * sin(d_lon / 2) * sin(d_lon / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        d = radius * c
        return (minutes_in_an_hour * d) / speed

    @staticmethod
    def calculate_tsp(number_of_nodes, time_matrix):
        node_range = range(number_of_nodes)
        dist = tsp.tsp(node_range, time_matrix, 10)[0]
        return dist
