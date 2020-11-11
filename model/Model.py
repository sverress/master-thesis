import gurobipy as gp
from gurobipy import GRB
from itertools import product
import pandas as pd
import os
from model.ModelInput import ModelInput


class Model:
    def __init__(self, input: ModelInput):
        """
        Formulation of mathematical problem in gurobi framework
        :param input: ModelInput object with input variables for the model
        """
        self.m = gp.Model("TOP")
        self._ = input

        # Cartesian Products
        self.cart_locs = list(product(self._.locations, repeat=2))
        self.cart_loc_v = list(product(self._.locations, self._.service_vehicles))
        self.cart_loc_loc_v = list(
            product(self._.locations, self._.locations, self._.service_vehicles)
        )

        # Init variables

        # x_ijv - 1 if, for service vehicle v, visit to location i is followed by a visit to location j- 0 otherwise
        self.x = self.m.addVars(self.cart_loc_loc_v, vtype=GRB.BINARY, name="x")
        # b_iv - 1 if location i is visited by service vehicle v- 0 otherwise
        self.y = self.m.addVars(self.cart_loc_v, vtype=GRB.BINARY, name="y")
        # p_iv - 1 if service vehicle v picks up a scooter at location i - 0 otherwise
        self.p = self.m.addVars(self.cart_loc_v, vtype=GRB.BINARY, name="p")
        # u_iv - position of location i for service vehicle v route
        self.u = self.m.addVars(self.cart_loc_v, vtype=GRB.INTEGER, name="u")
        # l_iv - load (number of scooters) when entering location i
        self.l = self.m.addVars(self.cart_loc_v, vtype=GRB.INTEGER, name="l")
        self.setup()

    def get_parameters(self):
        return self._

    def setup(self):
        self.set_objective()
        self.set_constraints()

    def set_objective(self):
        self.m.setObjective(
            gp.quicksum(self._.reward[i] * self.y[(i, v)] for i, v in self.cart_loc_v)
            - gp.quicksum(
                self._.reward[i] * self.p[(i, v)]
                for i, v in self.cart_loc_v
                if i in self._.scooters
            ),
            GRB.MAXIMIZE,
        )

    def set_constraints(self):
        # Add constraints (2): guarantee that each service vehicle starts and ends in at the depot.
        self.m.addConstr(
            gp.quicksum(
                self.x[(self._.depot, j, v)]
                for j, v in self.cart_loc_v
                if j != self._.depot
            )
            == self._.num_service_vehicles,
            "must_visit_depot_first",
        )
        self.m.addConstr(
            gp.quicksum(
                self.x[(i, self._.depot, v)]
                for i, v in self.cart_loc_v
                if i != self._.depot
            )
            == self._.num_service_vehicles,
            "must_visit_depot_end",
        )

        # Add constraints (3): ensure that every location is visited at most once.
        self.m.addConstrs(
            (
                gp.quicksum(self.y[(k, v)] for v in self._.service_vehicles) <= 1
                for k in self._.locations
                if k != self._.depot
            ),
            "only_one_visit_pr_scooter",
        )

        # Add constraints (4): ensure that each vehicle capacity is not exceeded
        self.m.addConstrs(
            (
                gp.quicksum(self.y[(k, v)] for k in self._.scooters) <= self._.Q_b[v]
                for v in self._.service_vehicles
            ),
            "battery_capacity",
        )

        # Add constraints (5): guarantee the connectivity of each service vehicle path
        self.m.addConstrs(
            (
                gp.quicksum(self.x[(i, k, v)] for i in self._.locations)
                == self.y[(k, v)]
                for k, v in self.cart_loc_v
            ),
            "connectivity_inn",
        )
        self.m.addConstrs(
            (
                gp.quicksum(self.x[(k, j, v)] for j in self._.locations)
                == self.y[(k, v)]
                for k, v in self.cart_loc_v
            ),
            "connectivity_out",
        )

        # Add constraints (6): ensure that the length of the paths does not exceed the shift
        self.m.addConstrs(
            (
                gp.quicksum(
                    self._.time_cost[(i, j)] * self.x[(i, j, v)]
                    for i, j in self.cart_locs
                )
                <= self._.T_max
                for v in self._.service_vehicles
            ),
            "time_constraints",
        )

        # Add constraints (7):
        self.m.addConstrs(
            (
                self.l[(i, v)]
                + self.p[(i, v)]
                - self.l[(j, v)]
                - self._.Q_s[v] * (1 - self.x[(i, j, v)])
                <= 0
                for i, j, v in self.cart_loc_loc_v
                if i in self._.scooters and j != i
            ),
            "vehicle_capacity_pick_up_less",
        )

        # Add constraints (8):
        self.m.addConstrs(
            (
                self.l[(i, v)]
                + self.p[(i, v)]
                - self.l[(j, v)]
                + self._.Q_s[v] * (1 - self.x[(i, j, v)])
                >= 0
                for i, j, v in self.cart_loc_loc_v
                if i in self._.scooters and j != i
            ),
            "vehicle_capacity_pick_up_greater",
        )

        # Add constraints (9):
        self.m.addConstrs(
            (
                self.l[(i, v)]
                - self.y[(i, v)]
                - self.l[(j, v)]
                - self._.Q_s[v] * (1 - self.x[(i, j, v)])
                <= 0
                for i, j, v in self.cart_loc_loc_v
                if i in self._.delivery and j != i
            ),
            "vehicle_capacity_delivery_less",
        )

        # Add constraints (10):
        self.m.addConstrs(
            (
                self.l[(i, v)]
                - self.y[(i, v)]
                - self.l[(j, v)]
                + self._.Q_s[v] * (1 - self.x[(i, j, v)])
                >= 0
                for i, j, v in self.cart_loc_loc_v
                if i in self._.delivery and j != i
            ),
            "vehicle_capacity_delivery_greater",
        )

        # Add constraints (11):
        self.m.addConstrs(
            (0 <= self.l[(i, v)] for i, v in self.cart_loc_v if i != self._.depot),
            "vehicle_capacity_cap_noneg",
        )
        self.m.addConstrs(
            (
                self.l[(i, v)] <= self._.Q_b[v]
                for i, v in self.cart_loc_v
                if i != self._.depot
            ),
            "vehicle_capacity_cap",
        )

        # Add constraints (12):
        self.m.addConstrs(
            (self.l[(0, v)] == 0 for v in self._.service_vehicles),
            "vehicle_capacity_depot_in",
        )

        # Add constraints (13):
        self.m.addConstrs(
            (
                self.l[(i, v)] - self._.Q_s[v] * (1 - self.x[(0, i, v)]) <= 0
                for i, v in self.cart_loc_v
                if i != self._.depot
            ),
            "vehicle_capacity_depot_out",
        )

        # Add constraints (14):
        self.m.addConstrs(
            (2 <= self.u[(i, v)] for i, v in self.cart_loc_v if i != self._.depot),
            "subtours_1",
        )
        self.m.addConstrs(
            (
                self.u[(i, v)] <= self._.num_locations
                for i, v in self.cart_loc_v
                if i != self._.depot
            ),
            "subtours_2",
        )

        # Add constraints (15):
        self.m.addConstrs(
            (
                self.u[i, v] - self.u[j, v] + 1
                <= (self._.num_locations - 1) * (1 - self.x[i, j, v])
                for i, j, v in self.cart_loc_loc_v
                if i != self._.depot
            ),
            "subtours_3",
        )

    def optimize_model(self):
        self.m.optimize()

    def print_solution(self):
        # Print solution
        for v in self.m.getVars():
            if v.x > 0:
                print(f"{v.varName}: {int(v.x)}")
        print(f"Obj: {self.m.objVal}")

        print(f"Obj: {self.m.objVal}")

    def print_model(self, delete_file=True):
        self.m.write("model.lp")
        with open("model.lp") as f:
            for line in f.readlines():
                print(line)
        if delete_file:
            os.remove("model.lp")


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
    model = Model(ModelInput(scooters, delivery, depot, service_vehicles))
    model.optimize_model()
    model.print_solution()
