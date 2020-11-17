import errno
import os
import pandas as pd
import math

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
        model_class,
    ):
        """
        Wrapper class for a Model class. Contains both raw input data and model input data

        :param scooters: dataframe with lat lon and battery in %
        :param delivery_nodes: dataframe with lat lon of delivery nodes
        :param depot: (lat, lon)
        :param service_vehicles: dict - ["type"]: (#numbers, scooter capacity, battery capacity)
        :param number_of_sections: int number_of_sections
        :param T_max: int - time limit for vehicles
        :param computational_limit: int - max solution time for model
        :param bound: tuple that defines the bound of the geographical area in play
        """

        # Save raw data
        self.scooters = scooters
        self.delivery_nodes = delivery_nodes
        self.depot = depot
        self.service_vehicles = service_vehicles
        self.T_max = T_max
        self.computational_limit = computational_limit

        # Context
        self.bound = bound

        # Model
        self.model_input = model_class.get_input_class()(
            scooters, delivery_nodes, depot, service_vehicles, T_max
        )
        self.model = model_class(self.model_input, computational_limit)
        self.number_of_sections = number_of_sections

    def run(self):
        """
        Runs the model of the instance
        """
        self.model.optimize_model()

    def visualize_solution(self, save=False):
        """
        See documentation of visualize_solution function from visualization
        """
        visualize_solution(self, save)

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

    def save_model(self):
        """
        Function to save gurobi models, file name represents: zones per axis_nodes per zone_Tmax_#vehicles_computational limit
        """

        try:
            os.makedirs("saved_models")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        file_path_solution = "saved_models/" + self.get_model_name() + ".json"
        file_path_params = "saved_models/" + self.get_model_name() + ".sol"
        self.model.m.write(file_path_solution)
        self.model.m.write(file_path_params)

    def get_model_name(self):
        return "model_%d_%d_%d_%d_%d" % (
            self.number_of_sections,
            len(self.scooters) / (self.number_of_sections * 2),
            self.service_vehicles["car"][0] + self.service_vehicles["bike"][0],
            self.T_max,
            self.computational_limit,
        )

    def is_feasible(self):
        return self.model.m.MIPGap != math.inf
