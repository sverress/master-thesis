from Model import Model
from enum import Enum, auto


class Status(Enum):
    NO_MODEL = auto()
    READY_TO_RUN = auto()
    RUNNING = auto()
    FINISHED = auto()


class Instance:
    def __init__(self, model_input=None):
        if model_input:
            self.model = Model(model_input)
            self.status = Status.READY_TO_RUN
        else:
            self.model = None
            self.status = Status.NO_MODEL

    def run(self):
        if self.status.READY_TO_RUN:
            self.status = Status.RUNNING
            self.model.optimize_model()
            self.status = Status.FINISHED
            self.model.print_solution()
        else:
            raise ValueError("Not ready to run instance")

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
        self.create_node_dict()

    def visualize_graph(self):
        pass
