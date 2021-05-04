import copy

import classes
import decision
import decision.value_functions
import clustering.scripts
import errno
import pandas as pd
import os
import numpy as np
from openpyxl import load_workbook


def metrics_to_xlsx(instances: [classes.World]):

    parameter_name = instances[0].metrics.testing_parameter_name
    parameter_values_names = []
    metrics_data = []
    for instance in instances:
        parameter_values_names.append(str(instance.metrics.testing_parameter_value))
        metrics_data.append(instance.metrics.timeline)
        instance_metrics = instance.metrics.get_all_metrics()
        for metric in instance_metrics:
            metrics_data.append(metric)

    try:
        os.makedirs("computational_study")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if not os.path.isfile("computational_study/policy_evaluation.xlsx"):
        pd.DataFrame().to_excel("computational_study/policy_evaluation.xlsx")
    book = load_workbook("computational_study/policy_evaluation.xlsx")
    writer = pd.ExcelWriter(
        "computational_study/policy_evaluation.xlsx", engine="openpyxl"
    )
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    sheets = [ws.title.split("-")[0] for ws in book.worksheets]

    sheet_name = (
        f"{parameter_name.title()}-1"
        if parameter_name.title() not in sheets
        else f"{parameter_name.title()}-{sheets.count(str(parameter_name).title())+1}"
    )

    columns = pd.MultiIndex.from_product(
        [
            [parameter_name],
            parameter_values_names,
            ["Timeline", "Lost demand", "Avg. neg dev", "Def. Battery"],
        ]
    )

    df = pd.DataFrame(
        np.array(metrics_data).T, columns=columns
    )  # convert dict to dataframe

    df.to_excel(
        writer, sheet_name=sheet_name, startcol=1, startrow=1
    )  # write dataframe to file

    writer.save()


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
