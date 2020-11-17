"""
A file for testing stuff without dealing with circular imports
"""
import argparse
from instance.InstanceManager import InstanceManager
from model.StandardModel import StandardModel
from model.AlternativeModel import AlternativeModel


def run_model(model_class):
    manager = InstanceManager()
    instance = manager.create_test_instance(3, 10, 2, 20, 3, model_class)
    instance.visualize_raw_data_map()
    instance.run()
    instance.model.print_solution()
    instance.visualize_solution()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", help="type of model: either standard or alternative"
    )
    args = parser.parse_args()
    if args.model and args.model == "standard":
        run_model(StandardModel)
    if args.model and args.model == "alternative":
        run_model(AlternativeModel)
