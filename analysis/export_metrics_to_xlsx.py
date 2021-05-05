import copy

import classes
import decision
import decision.value_functions
import clustering.scripts
import errno
import pandas as pd
import os
from openpyxl import load_workbook
from personal import *


def metrics_to_xlsx(instances: [classes.World]):
    """
    Method to export metrics for a list of evaluated instances to Excel
    :param instances: list of instances to be exported
    """
    # assuming that all instances is tested on the same parameter
    parameter_name = instances[0].metrics.testing_parameter_name
    column_tuples = []
    metrics_data = []

    # loop through all instances and record their metrics
    for instance in instances:
        metrics_data, column_tuples = add_metric_column(
            instance, column_tuples=column_tuples, metrics_data=metrics_data
        )

    # creating the directory to save the file if it doesn't exist
    try:
        os.makedirs(PATH_COMPUTATIONAL_STUDY)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    file_name = f"{PATH_COMPUTATIONAL_STUDY}/{parameter_name.title()}.xlsx"

    # if the file isn't created -> create a new .xlsx file
    if not os.path.isfile(file_name):
        pd.DataFrame().to_excel(file_name)
    book = load_workbook(file_name)
    writer = pd.ExcelWriter(file_name, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    sheets = [ws.title.split("_")[0] for ws in book.worksheets]

    try:
        # have to replace : since its illegal that a sheet name contains the character
        instance_created_at = instances[0].created_at.replace(":", ".")

        sheet_name = (
            f"{instance_created_at}_1"
            if instance_created_at not in sheets
            else f"{instance_created_at}_{sheets.count(instance_created_at) + 1}"
        )

        columns = pd.MultiIndex.from_tuples(
            column_tuples,
            names=[
                "Parameter",
                "Parameter value",
                "Policy",
                "Shifts trained",
                "Metrics",
            ],
        )

        df = pd.DataFrame(
            pd.concat(metrics_data, axis=1, ignore_index=True).to_numpy(),
            columns=columns,
        )  # concatenate all metrics dataframes to one dataframe

        df.to_excel(
            writer, sheet_name=sheet_name, startcol=1, startrow=1
        )  # write dataframe to file

        # TODO - add a sheet with globals hyper-parameters

    finally:
        writer.save()


def add_metric_column(instance, metrics_data, column_tuples):
    metric_variables = instance.metrics.__dict__
    parameter_name = metric_variables["testing_parameter_name"]
    del metric_variables["testing_parameter_name"]
    parameter_value = metric_variables["testing_parameter_value"]
    del metric_variables["testing_parameter_value"]

    for i, metric in enumerate(metric_variables.keys()):
        column_tuples.append(
            (
                parameter_name.title(),
                str(parameter_value),
                instance.policy.__str__(),
                str(instance.policy.value_function.shifts_trained)
                if hasattr(instance.policy, "value_function")
                else "N.A",
                metric.replace("_", " ").title(),
            )
        )
        metrics_data.append(pd.DataFrame({i: metric_variables[metric]}))

    return metrics_data, column_tuples


def example_write_to_excel():
    # TODO: rewrite so that a cached and trained world is used
    POLICY = decision.EpsilonGreedyValueFunctionPolicy(
        decision.value_functions.LinearValueFunction()
    )
    world = classes.World(
        120,
        POLICY,
        clustering.scripts.get_initial_state(100, 10),
        test_parameter_name="Learning rate",
        test_parameter_value=0.9,
        visualize=False,
        verbose=False,
    )
    copy_world = copy.deepcopy(world)
    copy_world.metrics.testing_parameter_value = 0.95
    world.run()
    copy_world.run()
    metrics_to_xlsx([world, copy_world])


if __name__ == "__main__":
    example_write_to_excel()
