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
        service_vehicles: tuple,
        optimal_state: list,
        number_of_sections: int,
        number_of_zones: int,
        T_max: int,
        is_percent_t_max: bool,
        computational_limit: int,
        bound: tuple,
        model_class,
        **kwargs,
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
        self.is_percent_T_max = is_percent_t_max
        self.computational_limit = computational_limit
        self.optimal_state = optimal_state
        self.number_of_sections = number_of_sections
        self.number_of_zones = number_of_zones
        self.seed = kwargs.get("seed", None)

        # Context
        self.bound = bound

        # Model
        self.model_input = model_class.get_input_class()(
            scooters,
            delivery_nodes,
            depot,
            service_vehicles,
            optimal_state,
            T_max,
            is_percent_t_max,
            self.number_of_zones,
            theta=kwargs.get("theta", 0.05),
            beta=kwargs.get("beta", 0.8),
        )
        self.model = model_class(
            self.model_input,
            time_limit=computational_limit,
            valid_inequalities=kwargs.get("valid_inequalities", None),
            symmetry=kwargs.get("symmetry", None),
            subsets=kwargs.get("subsets", None),
        )

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

    def visualize_raw_data_map(
        self, model_name="", save=False, time_stamp=time.strftime("%d-%m %H.%M")
    ):
        """
       See documentation of visualize_solution function from visualization
       """
        visualize_test_instance(
            self.scooters,
            self.delivery_nodes,
            self.bound,
            self.number_of_sections,
            model_name,
            save,
            time_stamp,
        )

    def print_instance(self):
        print("\n-------------- INSTANCE --------------\n")
        print(self.get_model_name(True))
        print("\n--------------------------------------\n")

    def get_runtime(self):
        """
        :return: the elapsed time in seconds to get to optimal solution in gurobi
        """
        return self.model.m.Runtime

    def get_number_of_nodes(self):
        return len(self.scooters) + len(self.delivery_nodes) + 1

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
        visit_percentage = sum(visits) / (len(self.model_input.locations) - 1)

        data["Visit Percentage"] = visit_percentage
        data["Deviation Before"] = self.deviation_before()
        deviation_after, deviation_after_squared = self.deviation_from_optimal_state()
        data["Deviation After"] = deviation_after
        data["Deviation After Squared"] = deviation_after_squared
        data["Instance"] = self.instance_to_dict()
        data["Variables"] = self.get_model_variables()
        data["Seed"] = self.seed

        if self.model.valid_inequalities:
            data["Valid Inequalities"] = self.model.valid_inequalities
        else:
            data["Valid Inequalities"] = "-"
        if self.model.symmetry:
            data["Symmetry constraint"] = self.model.symmetry
        else:
            data["Symmetry constraint"] = "-"

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
            "T_max": self.model_input.shift_duration,
            "is_percent_T_max": (self.is_percent_T_max, self.T_max),
            "computational_limit": self.computational_limit,
            "bound": self.bound,
            "model_class": self.model.to_string(False),
            "theta": self.model_input.theta if self.model.to_string() == "A" else "x",
            "beta": self.model_input.beta if self.model.to_string() == "A" else "x",
        }

    def get_model_variables(self):
        variables = {}
        for var in self.model.m.getVars():
            variables[var.VarName] = var.X
        return variables

    def get_model_name(self, print_mode=False):
        num_of_service_vehicles, scooter_cap, battery_cap = self.service_vehicles
        scooters_per_section = int(len(self.scooters) / (self.number_of_sections * 2))
        if not self.model.symmetry:
            symmetry = "None"
        else:
            symmetry = self.model.symmetry
        if not self.model.valid_inequalities:
            valid_inequalities = "None"
        else:
            valid_inequalities = self.model.valid_inequalities
        if self.is_percent_T_max:
            percent = (
                f"T max calculated from TSP - {self.is_percent_T_max}\n"
                + f"Shift duration as percent of TSP - {int(self.T_max*100)}%\n"
            )
        else:
            percent = f"T max calculated from TSP - {self.is_percent_T_max}\n"

        return (
            f"Number of sections - {self.number_of_sections}\n"
            + f"Scooters per zone - {scooters_per_section}\n"
            + f"Number of service vehicles - {num_of_service_vehicles}\n"
            + percent
            + f"Shift duration - {int(self.model_input.shift_duration)}\n"
            + f"Computational limit - {self.computational_limit}\n"
            + f"Model type - {self.model.to_string(False)}\n"
            + f"Symmetry constraint - {' '.join(['%-2s,' % (x,) for x in symmetry])}\n"
            + f"Valid Inequalities - {' '.join(['%-2s,' % (x,) for x in valid_inequalities])}\n"
            + f"Seed - {self.seed}"
            if print_mode
            else f"model_{self.number_of_sections}_{scooters_per_section}_{num_of_service_vehicles}_"
            + f"{int(self.model_input.shift_duration)}_{self.computational_limit}_"
            + f"{self.model.to_string()}_{round(self.model_input.theta,4)}_{round(self.model_input.beta,3)}_"
            f"{symmetry}_{valid_inequalities}"
        )

    def is_feasible(self):
        return self.model.m.MIPGap != math.inf

    def deviation_from_optimal_state(self):
        optimal_state = self.calculate_optimal_state()
        deviation = 0
        deviation_squared = 0
        for z in self.model_input.zones:
            battery_in_zone = 0
            for s in self.model_input.zone_scooters[z]:
                battery = 0
                visited = False
                for v in self.model_input.service_vehicles:
                    if s in self.model_input.scooters and self.model.p[(s, v)].x == 1:
                        visited = True
                    elif self.model.y[(s, v)].x == 1:
                        visited = True
                        battery += 1
                if not visited and s in self.model_input.scooters:
                    battery += self.model_input.battery_level[s]

                battery_in_zone += battery

            deviation += abs(optimal_state - battery_in_zone)
            deviation_squared += (optimal_state - battery_in_zone) ** 2

        return deviation, deviation_squared

    def deviation_before(self):
        optimal_state = self.calculate_optimal_state()
        deviation = 0
        for z in self.model_input.zones:
            battery_in_zone = sum(
                [
                    self.model_input.battery_level[s]
                    for s in self.model_input.zone_scooters[z]
                    if s in self.model_input.scooters
                ]
            )
            deviation += abs(optimal_state - battery_in_zone)

        return deviation

    def calculate_optimal_state(self):
        return self.model_input.num_scooters / (self.number_of_sections ** 2)