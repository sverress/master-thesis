from gurobipy import GRB
import gurobipy as gp

from model.BaseModel import BaseModel
from model.BaseModelInput import BaseModelInput


class StandardModelInput(BaseModelInput):
    def compute_reward_matrix(self, scooter_list, delivery_nodes_list):
        return (
            [0.0]
            + [1 - x / 100 for x in scooter_list["battery"]]
            + [1.0] * len(delivery_nodes_list.index)
        )  # Reward for visiting location i (uniform distribution). Eks for 3 locations [0, 0.33, 0.66]


class StandardModel(BaseModel):
    def setup(self):
        super().setup()

    @staticmethod
    def get_input_class():
        return StandardModelInput

    def to_string(self, short_name=True):
        return "S" if short_name else "Standard"

    def set_objective(self):
        self.m.setObjective(
            gp.quicksum(
                self._.reward[i] * self.y[(i, v)] for i, v in self.cart_loc_v_not_depot
            )
            - gp.quicksum(
                self._.battery_level[i] * self.p[(i, v)]
                for i, v in self.cart_loc_v_scooters
            ),
            GRB.MAXIMIZE,
        )
