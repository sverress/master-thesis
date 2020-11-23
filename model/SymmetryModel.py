from model.StandardModel import StandardModel
import gurobipy as gp


class SymmetryModel(StandardModel):
    def setup(self):
        # Adding the constraints and call the set_objective function
        super().setup()
        # Adding symmetry constraints
        self.m.addConstrs((gp.quicksum(self.x[(i, j, v)] for i, j in self.cart_locs)))
