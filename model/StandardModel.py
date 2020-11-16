from gurobipy import GRB
import pandas as pd
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
    @staticmethod
    def get_input_class():
        return StandardModelInput

    def set_objective(self):
        self.m.setObjective(
            gp.quicksum(
                self._.reward[i] * self.y[(i, v)] for i, v in self.cart_loc_v_not_depot
            )
            - gp.quicksum(
                self._.reward[i] * self.p[(i, v)] for i, v in self.cart_loc_v_scooters
            ),
            GRB.MAXIMIZE,
        )


if __name__ == "__main__":
    scooters = pd.DataFrame(
        [
            [59.914928, 10.747932, 21.0],
            [59.913464, 10.732058, 53.0],
            [59.915516, 10.775063, 69.0],
            [59.932115, 10.712367, 10.0],
        ],
        columns=["lat", "lon", "battery"],
    )
    delivery = pd.DataFrame(
        [[59.937612, 10.785628], [59.922692, 10.728357]], columns=["lat", "lon"],
    )

    depot = (59.91151, 10.763182)
    service_vehicles = {"car": (1, 3, 10), "bike": (1, 0, 4)}
    model = StandardModel(BaseModelInput(scooters, delivery, depot, service_vehicles))
    model.optimize_model()
    model.print_solution()
