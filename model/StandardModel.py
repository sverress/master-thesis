from gurobipy import GRB
import pandas as pd
from model.ModelInput import ModelInput
import gurobipy as gp

from model.BaseModel import BaseModel


class StandardModel(BaseModel):
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
    model = StandardModel(ModelInput(scooters, delivery, depot, service_vehicles))
    model.optimize_model()
    model.print_solution()
