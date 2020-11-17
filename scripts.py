from instance.InstanceManager import InstanceManager
from instance.helpers import save_models_to_excel

if __name__ == "__main__":
    manager = InstanceManager()
    manager.create_multiple_instances()
    for instance_key in manager.instances.keys():
        instance = manager.get_instance(instance_key)
        instance.run()
        instance.visualize_solution()
        instance.save_model()
    save_models_to_excel()
