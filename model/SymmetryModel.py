from model import BaseModelInput
from model.StandardModel import StandardModel
import gurobipy as gp


class SymmetryModel(StandardModel):
    def __init__(self, model_input: BaseModelInput, **kwargs):
        self.constraint_type = "advanced"
        super().__init__(model_input, **kwargs)

    def get_constraints(self):
        return {
            "simple": [
                (
                    (
                        gp.quicksum(self.x[(i, j, v)] for i, j in self.cart_locs)
                        >= gp.quicksum(self.x[(i, j, v + 1)] for i, j in self.cart_locs)
                    )
                    for v in range(self._.num_service_vehicles - 1)
                )
            ],
            "advanced": [
                (
                    gp.quicksum(self.y[(i, v)] for v in range(i + 1)) <= 1
                    for i in self._.service_vehicles
                ),
                (
                    self.y[(i, v)]
                    <= gp.quicksum(
                        self.y[(p, s)]
                        for p in range(v - 1, i)
                        for s in range(v - 1, min(p, self._.num_service_vehicles))
                    )
                    for i in self._.locations
                    if i not in [0, 1]
                    for v in self._.service_vehicles
                    if v != 0
                ),
            ],
        }

    def setup(self):
        # Adding the constraints and call the set_objective function
        super().setup()
        # Adding symmetry constraints
        for i, constr in enumerate(self.get_constraints()[self.constraint_type]):
            self.m.addConstrs(constr, f"symmetry{i}")
