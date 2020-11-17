"""
A file for testing stuff without dealing with circular imports
"""
from instance.InstanceManager import InstanceManager
from model.StandardModel import StandardModel

if __name__ == "__main__":
    manager = InstanceManager()
    instance = manager.create_test_instance(2, 5, 2, 20, 50, StandardModel)
