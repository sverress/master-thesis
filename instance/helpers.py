import numpy as np
import pandas as pd
import json
import os
from itertools import product


def create_sections(number_of_sections: int, bound: tuple):
    """
    Generate lat lon coordinates for the boundaries of sections
    :param number_of_sections: integer
    :param bound: tuple of lat_min, lat_max, lon_min, lon_max
    :return: tuple(dataframe with lat lon as columns containing the boundaries of the sections,
    list of coordinates defining all zones generated)
    """
    lat_min, lat_max, lon_min, lon_max = bound
    lon_length = lon_max - lon_min
    lat_length = lat_max - lat_min
    section_limits = pd.DataFrame()
    epsilon = 0.000001  # Just a small number
    section_limits["lon"] = np.arange(
        lon_min, lon_max + epsilon, lon_length / number_of_sections
    )
    section_limits["lat"] = np.arange(
        lat_min, lat_max + epsilon, lat_length / number_of_sections
    )
    section_coordinates = []
    for j in range(1, number_of_sections + 1):
        for i in range(1, number_of_sections + 1):
            section_coordinates.append(
                (
                    section_limits["lat"][j - 1],
                    section_limits["lat"][j],
                    section_limits["lon"][i - 1],
                    section_limits["lon"][i],
                )
            )
    return section_limits, section_coordinates


def load_test_parameters_from_json():
    """
    Loads parameters for computational study from json file and makes all combinations of these.
    :return list - all combinations of parameters from json file
    """
    with open("instance/test_instances.json") as json_file:
        data = json.load(json_file)
    ranges = []
    for key in data:
        if type(data[key]) is list:
            parameter_min, parameter_max, parameter_increment = data[key]
            ranges.append(range(parameter_min, parameter_max + 1, parameter_increment))
        else:
            parameter = data[key]
            ranges.append(range(parameter, parameter + 1))

    return list(product(*ranges))


def save_models_to_excel():
    """
    Iterates through all saved models and store the information in an excel sheet.
    Saved information: zones, nodes per zone, # vehicles, T_max, solution time, GAP
    """
    zones, nodes_per_zone, number_of_vehicles, T_max, solution_time, gap = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for root, dirs, files in os.walk("saved models/", topdown=True):
        for file in files:
            with open(root + file) as file_path:
                model = json.load(file_path)
            model_param = file.split("_")
            zones.append(int(model_param[1]))
            nodes_per_zone.append(int(model_param[2]))
            number_of_vehicles.append(int(model_param[3]))
            T_max.append(int(model_param[4]))
            solution_time.append(float(model["SolutionInfo"]["Runtime"]))
            gap.append(float(model["SolutionInfo"]["MIPGap"]))

    df = pd.DataFrame(
        list(zip(zones, nodes_per_zone, number_of_vehicles, T_max, solution_time, gap)),
        columns=[
            "Zones",
            "Nodes per zone",
            "Number of vehicles",
            "T_max",
            "Solution time",
            "Gap",
        ],
    )
    df.to_excel(
        "computational study/comp_study_summary.xlsx",
        float_format="%.5f",
        sheet_name="Comp study",
    )


def get_center_of_bound(bound):
    lat_min, lat_max, lon_min, lon_max = bound
    return lat_min + (lat_max - lat_min) / 2, lon_min + (lon_max - lon_min) / 2
