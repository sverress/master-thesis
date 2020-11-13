from model.Model import Model, ModelInput
import pandas as pd
from visualization.visualizer import (
    visualize_solution,
    visualize_test_instance,
)


class Instance:
    def __init__(
        self,
        scooters: pd.DataFrame,
        delivery_nodes: pd.DataFrame,
        depot: tuple,
        service_vehicles: dict,
        number_of_sections: int,
        T_max: int,
        computational_limit: int,
        bound: tuple,
    ):
        """
        Wrapper class for a Model class. Contains both raw input data and model input data

        :param scooters: dataframe with lat lon and battery in %
        :param delivery_nodes: dataframe with lat lon of delivery nodes
        :param depot: (lat, lon)
        :param service_vehicles: dict - ["type"]: (#numbers, scooter capacity, battery capacity)
        :param number_of_sections: int number_of_sections
        :param bound: tuple that defines the bound of the geographical area in play
        """

        # Save raw data
        self.scooters = scooters
        self.delivery_nodes = delivery_nodes
        self.depot = depot
        self.service_vehicles = service_vehicles

        # Context
        self.bound = bound

        # Model
        self.model_input = ModelInput(
            scooters, delivery_nodes, depot, service_vehicles, T_max
        )
        self.model = Model(self.model_input, computational_limit)
        self.number_of_sections = number_of_sections

    def run(self):
        """
        Runs the model of the instance
        """
        self.model.optimize_model()

    def visualize_solution(self):
        """
        See documentation of visualize_solution function from visualization
        """
        visualize_solution(self)

    def visualize_raw_data_map(self):
        """
       See documentation of visualize_solution function from visualization
       """
        visualize_test_instance(
            self.scooters, self.delivery_nodes, self.bound, self.number_of_sections
        )

    def get_runtime(self):
        """
        :return: the elapsed time in seconds to get to optimal solution in gurobi
        """
        return self.model.m.Runtime
