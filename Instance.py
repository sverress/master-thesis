from Model import ModelInput, Model
from solution_visualizer import visualize_solution
import pandas as pd


class Instance:
    def __init__(self, scooters, delivery_nodes, depot, service_vehicles):
        self.locations = (
            [depot]
            + list(zip(scooters["lat"], scooters["lon"]))
            + list(zip(delivery_nodes["lat"], delivery_nodes["lon"]))
        )
        self.scooters = scooters
        self.delivery_nodes = delivery_nodes
        self.depot = depot
        self.service_vehicles = service_vehicles
        self.num_scooters = len(scooters.index)
        self.num_service_vehicles = sum(x[0] for x in service_vehicles.values())
        self.model = Model(
            ModelInput(scooters, delivery_nodes, depot, service_vehicles)
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
        visualize_solution(self)

    def visualize_graph(self):
        pass


if __name__ == "__main__":
    s = pd.DataFrame(
        [
            [59.914928, 10.747932, 21.0],
            [59.913464, 10.732058, 53.0],
            [59.915516, 10.775063, 69.0],
            [59.932115, 10.712367, 10.0],
        ],
        columns=["lat", "lon", "battery"],
    )
    d = pd.DataFrame(
        [[59.937612, 10.785628], [59.922692, 10.728357]], columns=["lat", "lon"],
    )

    depot = (59.91151, 10.763182)
    sv = {"car": (2, 3, 10), "bike": (0, 0, 4)}
    instance = Instance(s, d, depot, sv)
    instance.run()
    instance.visualize_solution()
