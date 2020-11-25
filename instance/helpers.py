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
    param = data["ranges"]
    for key in param:
        if type(param[key]) is list:
            parameter_min, parameter_max, parameter_increment = param[key]
            if (
                type(parameter_min) is float
                or type(parameter_max) is float
                or type(parameter_increment) is float
            ):
                ranges.append(
                    np.arange(
                        parameter_min, (parameter_max + 0.0001), parameter_increment,
                    )
                )
            else:
                ranges.append(
                    range(parameter_min, parameter_max + 1, parameter_increment)
                )
        elif type(param[key]) is float:
            parameter = param[key]
            ranges.append([parameter])
        else:
            parameter = param[key]
            ranges.append(range(parameter, parameter + 1))

    constraints = data["constraints"]
    constraint_list = []
    previous_list = []
    for constraint_name in constraints:
        constraint = constraints[constraint_name]
        constraint_list.append(constraint)
        # if type(constraint) is not list:
        #    constraint_list = list(product(*previous_list, (constraint,)))
        # else:
        #    constraint_list = list(product(*previous_list, constraint))
        # previous_list = constraint_list

    models = data["model"]["model_type"]
    if type(models) is not list:
        # range_model_list = list(product(*ranges, (models,)))
        input_list = list(product(*ranges, (models,), *constraint_list))
        # range_model_list = list(product(*ranges, models))
    else:
        input_list = list(product(*ranges, models, *constraint_list))

    instance_list = []
    for (
        zones_per_axis,
        nodes_per_zone,
        number_of_vehicles,
        T_max,
        time_limit,
        model_type,
        subtour,
        symmetry,
    ) in input_list:
        instance_list.append(
            {
                "number_of_sections": zones_per_axis,
                "number_of_scooters_per_section": nodes_per_zone,
                "number_of_vehicles": number_of_vehicles,
                "T_max": T_max,
                "time_limit": time_limit,
                "model_type": model_type,
                "subtour": subtour,
                "symmetry": symmetry,
                "T_max_is_percentage": data["model"]["T_max_is_percentage"],
            }
        )
    return instance_list


def save_models_to_excel(timestamp=time.strftime("%d-%m %H.%M")):
    """
    Iterates through all saved_models and store the information in an excel sheet.
    Saved information: zones, nodes per zone, # vehicles, T_max, solution time, GAP
    """
    try:
        os.makedirs("computational_study")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not os.path.isfile("computational_study/comp_study_summary.xlsx"):
        pd.DataFrame().to_excel("computational_study/comp_study_summary.xlsx")
    book = load_workbook("computational_study/comp_study_summary.xlsx")
    writer = pd.ExcelWriter(
        "computational_study/comp_study_summary.xlsx", engine="openpyxl"
    )
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    sheets = [ws.title for ws in book.worksheets]

    (
        zones,
        nodes_per_zone,
        number_of_vehicles,
        T_max,
        percent_t_max,
        solution_time,
        gap,
        visit_list,
        objective_value,
        model_type,
        deviation_before,
        deviation_after,
        symmetry_cons,
        subtour_cons,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])
    for root, dirs, files in os.walk("saved_models/", topdown=True):
        if len(dirs) == 0:
            if root.split("/")[-1] not in sheets:
                for file in files:
                    with open(f"{root}/{file}") as file_path:
                        model = json.load(file_path)
                    model_param = file.split("_")
                    zones.append(int(model_param[1]))
                    nodes_per_zone.append(int(model_param[2]))
                    number_of_vehicles.append(int(model_param[3]))
                    T_max.append(int(model_param[4]))
                    visit_list.append(model["Visit Percentage"])
                    deviation_before.append(model["Deviation Before"])
                    deviation_after.append(model["Deviation After"])
                    solution_time.append(float(model["SolutionInfo"]["Runtime"]))
                    gap.append(float(model["SolutionInfo"]["MIPGap"]))
                    objective_value.append(float(model["SolutionInfo"]["ObjVal"]))
                    model_type.append(model["Instance"]["model_class"])
                    symmetry_cons.append(model["Symmetry Constraint"])
                    subtour_cons.append(model["Subtour in set"])

                    is_percent_T_max, percent = model["Instance"]["is_percent_T_max"]
                    if is_percent_T_max:
                        percent_t_max.append(percent)
                    else:
                        percent_t_max.append(0)

    df = pd.DataFrame(
        list(
            zip(
                zones,
                nodes_per_zone,
                number_of_vehicles,
                T_max,
                percent_t_max,
                solution_time,
                gap,
                visit_list,
                deviation_before,
                deviation_after,
                objective_value,
                model_type,
                symmetry_cons,
                subtour_cons,
            )
        ),
        columns=[
            "Zones",
            "Nodes per zone",
            "Number of vehicles",
            "T_max",
            "Percent T_max",
            "Solution time",
            "Gap",
            "Visit percent",
            "Deviation before",
            "Deviation after",
            "Obj value",
            "Model type",
            "Symmetry Constraint",
            "Subtour in set",
        ],
    )

    df.to_excel(
        writer, float_format="%.5f", sheet_name=str(timestamp),
    )
    writer.save()


def get_center_of_bound(bound):
    lat_min, lat_max, lon_min, lon_max = bound
    return lat_min + (lat_max - lat_min) / 2, lon_min + (lon_max - lon_min) / 2
