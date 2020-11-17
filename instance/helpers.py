import csv
import errno
import numpy as np
import pandas as pd
import json
import os
import time
from itertools import product
from openpyxl import load_workbook


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
    Loads parameters for computational_study from json file and makes all combinations of these.
    :return list - all combinations of parameters from json file
    """
    with open("instance/test_instances.json") as json_file:
        data = json.load(json_file)
    ranges = []
    for key in data["ranges"]:
        if type(data[key]) is list:
            parameter_min, parameter_max, parameter_increment = data[key]
            ranges.append(range(parameter_min, parameter_max + 1, parameter_increment))
        else:
            parameter = data[key]
            ranges.append(range(parameter, parameter + 1))

    range_list = list(product(*ranges))

    instance_list = []
    for (
        zones_per_axis,
        nodes_per_zone,
        number_of_vehicles,
        T_max,
        time_limit,
    ) in range_list:
        instance_list.append(
            {
                "zones_per_axis": zones_per_axis,
                "nodes_per_zone": nodes_per_zone,
                "number_of_vehicles": number_of_vehicles,
                "T_max": T_max,
                "time_limit": time_limit,
                "model_type": data["model"]["model_type"],
                "T_max_is_percentage": data["model"]["T_max_is_percentage"],
            }
        )
    return instance_list


def save_models_to_excel():
    """
    Iterates through all saved_models and store the information in an excel sheet.
    Saved information: zones, nodes per zone, # vehicles, T_max, solution time, GAP
    """
    (
        zones,
        nodes_per_zone,
        number_of_vehicles,
        T_max,
        solution_time,
        gap,
        visit_list,
        objective_value,
    ) = ([], [], [], [], [], [], [], [])
    for root, dirs, files in os.walk("saved_models/", topdown=True):
        for file in files:
            if file.endswith(".json"):
                with open(root + file) as file_path:
                    model = json.load(file_path)
                model_param = file.split("_")
                zones.append(int(model_param[1]))
                nodes_per_zone.append(int(model_param[2]))
                number_of_vehicles.append(int(model_param[3]))
                T_max.append(int(model_param[4]))
                solution_time.append(float(model["SolutionInfo"]["Runtime"]))
                gap.append(float(model["SolutionInfo"]["MIPGap"]))
            if file.endswith(".sol"):
                with open(root + file) as csv_file:
                    reader = csv.reader(
                        (line.replace("  ", " ") for line in csv_file), delimiter=" "
                    )
                    next(reader)  # skip header
                    objective_value.append(float(next(reader)[4]))
                    y_list = []
                    for var, value in reader:
                        if var.startswith("y") and var[2] != "0":
                            y_list.append(int(value))
                visit_list.append(sum(y_list) / len(y_list))

    df = pd.DataFrame(
        list(
            zip(
                zones,
                nodes_per_zone,
                number_of_vehicles,
                T_max,
                solution_time,
                gap,
                visit_list,
                objective_value,
            )
        ),
        columns=[
            "Zones",
            "Nodes per zone",
            "Number of vehicles",
            "T_max",
            "Solution time",
            "Gap",
            "Visit Percentage",
            "Obj value",
        ],
    )

    try:
        os.makedirs("computational_study")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    book = load_workbook("computational_study/comp_study_summary.xlsx")
    writer = pd.ExcelWriter(
        "computational_study/comp_study_summary.xlsx", engine="openpyxl"
    )
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df.to_excel(
        writer, float_format="%.5f", sheet_name=str(time.strftime("%d-%m %H.%M")),
    )
    writer.save()
