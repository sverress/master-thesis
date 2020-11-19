import errno
import os
import pandas as pd
import math
import json
import time

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
        optimal_state: list,
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
        :param optimal_state: list of optimal state for each zone of the problem
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
        self.optimal_state = optimal_state

        # Context
        self.bound = bound

        # Model
        self.model_input = model_class.get_input_class()(
            scooters, delivery_nodes, depot, service_vehicles, optimal_state, T_max
        )
        self.model = model_class(self.model_input, time_limit=computational_limit)
        self.number_of_sections = number_of_sections

    def run(self):
        """
        Runs the model of the instance
        """
        self.model.optimize_model()

    def visualize_solution(
        self, save=False, edge_plot=False, time_stamp=time.strftime("%d-%m %H.%M")
    ):
        """
        See documentation of visualize_solution function from visualization
        """
        visualize_solution(self, save, edge_plot, time_stamp)

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

    def save_model_and_instance(self, time_stamp):
        """
        Function to save gurobi models, file name represents: zones per axis_nodes per zone_Tmax_#vehicles_computational limit
        """
        path = "saved_models/" + time_stamp
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        file_path_solution = f"{path}/{self.get_model_name()}.json"
        self.model.m.write(file_path_solution)

        with open(file_path_solution, "r") as jsonFile:
            data = json.load(jsonFile)

        visits = [self.model.y[key].x for key in self.model.y if 0 < key[0]]
        visit_percentage = sum(visits) / len(visits)

        data["Visit Percentage"] = visit_percentage
        data["Instance"] = self.instance_to_dict()
        data["Variables"] = self.get_model_variables()

        with open(file_path_solution, "w") as jsonFile:
            json.dump(data, jsonFile)

    def instance_to_dict(self):
        return {
            "scooters": self.scooters.to_dict(),
            "delivery_nodes": self.delivery_nodes.to_dict(),
            "depot": self.depot,
            "service_vehicles": self.service_vehicles,
            "optimal_state": self.optimal_state,
            "number_of_sections": self.number_of_sections,
            "T_max": self.T_max,
            "computational_limit": self.computational_limit,
            "bound": self.bound,
            "model_class": self.model.__class__.__name__,
        }

    def get_model_variables(self):
        variables = {}
        for var in self.model.m.getVars():
            variables[var.VarName] = var.X
        return variables

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
