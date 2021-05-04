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

    parameter_name = instances[0].metrics.testing_parameter_name
    parameter_values = []
    metrics_data = []
    for instance in instances:
        parameter_values.append(str(instance.metrics.testing_parameter_value))
        metrics_data.append(pd.DataFrame({"Timeline": instance.metrics.timeline}))
        instance_metrics = instance.metrics.get_all_metrics()
        for i, metric in enumerate(instance_metrics):
            metrics_data.append(pd.DataFrame({i: metric}))

    try:
        os.makedirs(PATH_COMPUTATIONAL_STUDY)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    file_name = f"{PATH_COMPUTATIONAL_STUDY}/{parameter_name.title()}.xlsx"

    if not os.path.isfile(file_name):
        pd.DataFrame().to_excel(file_name)
    book = load_workbook(file_name)
    writer = pd.ExcelWriter(file_name, engine="openpyxl")
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    sheets = [ws.title.split("_")[0] for ws in book.worksheets]

    instance_created_at = instances[0].created_at.replace(":", ".")
    # Sheet name can parameter_name + world.created_at
    sheet_name = (
        f"{instance_created_at}_1"
        if instance_created_at not in sheets
        else f"{instance_created_at}_{sheets.count(instance_created_at)+1}"
    )

    columns = pd.MultiIndex.from_product(
        [
            [parameter_name],
            parameter_values,
            ["Timeline", "Lost demand", "Avg. neg dev", "Def. Battery"],
        ]
    )

    df = pd.DataFrame(
        pd.concat(metrics_data, axis=1, ignore_index=True).to_numpy(), columns=columns
    )  # concatenate all metrics dataframes to one dataframe

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
