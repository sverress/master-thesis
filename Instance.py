from Model import ModelInput, Model
from solution_visualizer import visualize_solution


class Instance:
    def __init__(self, locations, num_scooters, num_service_vehicles):
        self.locations = locations
        self.num_scooters = num_scooters
        self.num_service_vehicles = num_service_vehicles
        self.model = Model(
            ModelInput(locations_coordinates, num_scooters, num_service_vehicles)
        )

    def run(self):
        self.model.optimize_model()
        # self.model.print_solution()

    def get_label(self, i):
        if i == 0:
            return "Depot"
        if 0 < i <= self.num_scooters:
            return "S"
        else:
            return "D"

    def create_node_dict(self):
        output = {}
        for i, index in enumerate(self.locations):
            output[index] = {"label": self.get_label(i)}
        return output

    def visualize_solution(self):
        visualize_solution(self, self.create_node_dict())

    def visualize_graph(self):
        pass


if __name__ == "__main__":
    locations_coordinates = [
        (0, 0),
        (1, 5),
        (3, 3),
        (2, 4),
        (3, 1),
        (1, 3),
    ]  # First element is depot

    # Constants
    number_of_scooters = 3  # Number of scooters
    number_of_service_vehicles = 2  # Number of service vehicles
    instance = Instance(
        locations_coordinates, number_of_scooters, number_of_service_vehicles
    )
    instance.run()
    instance.visualize_solution()
