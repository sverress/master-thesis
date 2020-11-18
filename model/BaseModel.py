import gurobipy as gp
from gurobipy import GRB
from itertools import product
import os
from abc import ABC, abstractmethod

from model.BaseModelInput import BaseModelInput


class BaseModel(ABC):
    def __init__(self, model_input: BaseModelInput, setup=True, time_limit=None):
        """
        Formulation of mathematical problem in gurobi framework
        :param model_input: ModelInput object with input variables for the model
        """
        self.m = gp.Model("TOP")
        self._ = model_input

        # Setting a computational time limit for the model
        if time_limit:
            self.m.Params.TimeLimit = time_limit

        # Cartesian Products
        self.cart_locs = list(product(self._.locations, repeat=2))
        self.cart_loc_v = list(product(self._.locations, self._.service_vehicles))
        self.cart_loc_loc_v = list(
            product(self._.locations, self._.locations, self._.service_vehicles)
        )
        self.cart_loc_v_not_depot = list(
            product(
                [loc for loc in self._.locations if loc != self._.depot],
                self._.service_vehicles,
            )
        )
        self.cart_loc_v_scooters = list(
            product(
                [loc for loc in self._.locations if loc in self._.scooters],
                self._.service_vehicles,
            )
        )

        # Init variables

        # x_ijv - 1 if, for service vehicle v, visit to location i is followed by a visit to location j- 0 otherwise
        self.x = self.m.addVars(self.cart_loc_loc_v, vtype=GRB.BINARY, name="x")
        # y_iv - 1 if location i is visited by service vehicle v- 0 otherwise
        self.y = self.m.addVars(self.cart_loc_v, vtype=GRB.BINARY, name="y")
        # p_iv - 1 if service vehicle v picks up a scooter at location i - 0 otherwise
        self.p = self.m.addVars(self.cart_loc_v_scooters, vtype=GRB.BINARY, name="p")
        # u_iv - position of location i for service vehicle v route
        self.u = self.m.addVars(self.cart_loc_v_not_depot, vtype=GRB.INTEGER, name="u")
        # l_iv - load (number of scooters) when entering location i
        self.l = self.m.addVars(self.cart_loc_v, vtype=GRB.INTEGER, name="l")
        if setup:
            self.setup()

    def get_parameters(self):
        return self._

    def setup(self):
        self.set_objective()
        self.set_constraints()

    @abstractmethod
    def set_objective(self):
        pass

    @staticmethod
    @abstractmethod
    def get_input_class():
        pass

    def set_constraints(self):
        #  guarantee that each service vehicle starts and ends in at the depot.
        self.m.addConstr(
            gp.quicksum(
                self.x[(self._.depot, j, v)] for j, v in self.cart_loc_v_not_depot
            )
            == self._.num_service_vehicles,
            "must_visit_depot_first",
        )
        self.m.addConstr(
            gp.quicksum(
                self.x[(i, self._.depot, v)] for i, v in self.cart_loc_v_not_depot
            )
            == self._.num_service_vehicles,
            "must_visit_depot_end",
        )

        #  Ensure that every location is visited at most once.
        self.m.addConstrs(
            (
                gp.quicksum(self.y[(k, v)] for v in self._.service_vehicles) <= 1
                for k in self._.locations
                if k != self._.depot
            ),
            "only_one_visit_pr_scooter",
        )

        #  Ensure that each vehicle capacity is not exceeded
        self.m.addConstrs(
            (
                gp.quicksum(self.y[(k, v)] for k in self._.scooters) <= self._.Q_b[v]
                for v in self._.service_vehicles
            ),
            "battery_capacity",
        )

        #  guarantee the connectivity of each service vehicle path
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

        #  Ensure that the length of the paths does not exceed the shift
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

        # Ensure that no scooters can be picked up in a demand zone
        self.m.addConstrs(
            (
                gp.quicksum(
                    self.p[(i, v)]
                    for i in self._.L_z[z]
                    if i in self._.scooters
                    for v in self._.service_vehicles
                )
                == 0
                for z in self._.Z_demand
            )
        )

        # Scooter capacity management
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

        self.m.addConstrs(
            (0 <= self.l[(i, v)] for i, v in self.cart_loc_v_not_depot),
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

        self.m.addConstrs(
            (self.l[(0, v)] == 0 for v in self._.service_vehicles),
            "vehicle_capacity_depot_in",
        )

        self.m.addConstrs(
            (
                self.l[(i, v)] - self._.Q_s[v] * (1 - self.x[(0, i, v)]) <= 0
                for i, v in self.cart_loc_v_not_depot
            ),
            "vehicle_capacity_depot_out",
        )

        # Subtour elimination
        self.m.addConstrs(
            (2 <= self.u[(i, v)] for i, v in self.cart_loc_v_not_depot), "subtours_1",
        )
        self.m.addConstrs(
            (
                self.u[(i, v)] <= self._.num_locations
                for i, v in self.cart_loc_v_not_depot
            ),
            "subtours_2",
        )

        self.m.addConstrs(
            (
                self.u[i, v] - self.u[j, v] + 1
                <= (self._.num_locations - 1) * (1 - self.x[i, j, v])
                for i, j, v in self.cart_loc_loc_v
                if i != self._.depot and j != self._.depot
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
