from model.Model import Model, ModelInput
from visualization.solution_visualizer import (
    visualize_solution,
    visualize_test_instance,
)


class Instance:
    def __init__(
        self,
        scooters,
        delivery_nodes,
        depot,
        service_vehicles,
        number_of_sections,
        bound,
    ):
        # TODO: Needs documentation

        # Save raw data
        self.scooters = scooters
        self.delivery_nodes = delivery_nodes
        self.depot = depot
        self.service_vehicles = service_vehicles

        # Context
        self.bound = bound

        # Model
        self.model_input = ModelInput(scooters, delivery_nodes, depot, service_vehicles)
        self.model = Model(self.model_input)
        self.number_of_sections = number_of_sections

    def run(self):
        # TODO: Needs documentation
        self.model.optimize_model()

    def get_label(self, i):
        # TODO: should be moved to solution visualization script
        if i == 0:
            return "Depot"
        if 0 < i <= self.model_input.num_scooters:
            return "S"
        else:
            return "D"

    def create_node_dict(self):
        # TODO: should be moved to solution visualization script. this should return a list. Needs documentation
        output = {}
        for i, index in enumerate(self.model_input.locations):
            output[index] = {"label": self.get_label(i)}
        return output

    def visualize_solution(self):
        # TODO: Needs documentation
        visualize_solution(self)

    def visualize_raw_data_map(self):
        visualize_test_instance(
            self.scooters, self.delivery_nodes, self.bound, self.number_of_sections
        )
