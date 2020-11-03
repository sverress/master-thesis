import gurobipy as gp
from gurobipy import GRB
import helpers
from itertools import product

locations_coordinates = [
    (0, 0),
    (1, 5),
    (3, 3),
    (2, 4),
    (3, 1),
    (1, 3),
]  # First element is depot

# Constants
num_scooters = 3  # Number of scooters
num_service_vehicles = 1  # Number of service vehicles
num_locations = len(locations_coordinates)

# Sets
locations = range(num_locations)
scooters = locations[1 : num_scooters + 1]
delivery = locations[num_scooters + 1 :]
service_vehicles = range(num_service_vehicles)
depot = 0

# Cartesian Products
cart_locs = list(product(locations, repeat=2))
cart_loc_v = list(product(locations, service_vehicles))
cart_loc_loc_v = list(product(locations, locations, service_vehicles))

# Parameters
reward = (
    [0.0]
    + [(i + 1) / num_scooters - 0.1 for i in range(num_scooters)]
    + [1.0] * (num_locations - num_scooters - 1)
)  # Reward for visiting location i (uniform distribution). Eks for 3 locations [0, 0.33, 0.66]
time_cost = {
    (i, j): helpers.compute_distance(locations_coordinates[i], locations_coordinates[j])
    for i, j in cart_locs
}  # Calculate distance in time between all locations
T_max = 15  # Duration of shift
Q_b = [5] * num_service_vehicles  # Battery capacity of service vehicle v
Q_s = [2] * num_service_vehicles


class Model:
    def __init__(self):
        self.m = gp.Model("TOP")

        # Init variables

        # x_ijv - 1 if, for service vehicle v, visit to location i is followed by a visit to location j- 0 otherwise
        self.x = self.m.addVars(cart_loc_loc_v, vtype=GRB.BINARY, name="x")
        # b_iv - 1 if location i is visited by service vehicle v- 0 otherwise
        self.y = self.m.addVars(cart_loc_v, vtype=GRB.BINARY, name="y")
        # p_iv - 1 if service vehicle v picks up a scooter at location i - 0 otherwise
        self.p = self.m.addVars(cart_loc_v, vtype=GRB.BINARY, name="p")
        # u_iv - position of location i for service vehicle v route
        self.u = self.m.addVars(cart_loc_v, vtype=GRB.INTEGER, name="u")
        # l_iv - load (number of scooters) when entering location i
        self.l = self.m.addVars(cart_loc_v, vtype=GRB.INTEGER, name="l")

    def setup(self):
        self.set_objective()
        self.set_constraints()

    def set_objective(self):
        self.m.setObjective(
            gp.quicksum(reward[i] * self.y[(i, v)] for i, v in cart_loc_v)
            - gp.quicksum(
                reward[i] * self.p[(i, v)] for i, v in cart_loc_v if i in scooters
            ),
            GRB.MAXIMIZE,
        )

    def set_constraints(self):
        # Add constraints (2): guarantee that each service vehicle starts and ends in at the depot.
        self.m.addConstr(
            gp.quicksum(self.x[(depot, j, v)] for j, v in cart_loc_v if j != depot)
            == num_service_vehicles,
            "must_visit_depot_first",
        )
        self.m.addConstr(
            gp.quicksum(self.x[(i, depot, v)] for i, v in cart_loc_v if i != depot)
            == num_service_vehicles,
            "must_visit_depot_end",
        )

        # Add constraints (3): ensure that every location is visited at most once.
        self.m.addConstrs(
            (
                gp.quicksum(self.y[(k, v)] for v in service_vehicles) <= 1
                for k in locations
                if k != depot
            ),
            "only_one_visit_pr_scooter",
        )

        # Add constraints (4): ensure that each vehicle capacity is not exceeded
        self.m.addConstrs(
            (
                gp.quicksum(self.y[(k, v)] for k in scooters) <= Q_b[v]
                for v in service_vehicles
            ),
            "battery_capacity",
        )

        # Add constraints (5): guarantee the connectivity of each service vehicle path
        self.m.addConstrs(
            (
                gp.quicksum(self.x[(i, k, v)] for i in locations) == self.y[(k, v)]
                for k, v in cart_loc_v
            ),
            "connectivity_inn",
        )
        self.m.addConstrs(
            (
                gp.quicksum(self.x[(k, j, v)] for j in locations) == self.y[(k, v)]
                for k, v in cart_loc_v
            ),
            "connectivity_out",
        )

        # Add constraints (6): ensure that the length of the paths does not exceed the shift
        self.m.addConstrs(
            (
                gp.quicksum(time_cost[(i, j)] * self.x[(i, j, v)] for i, j in cart_locs)
                <= T_max
                for v in service_vehicles
            ),
            "time_constraints",
        )

        # Add constraints (7):
        self.m.addConstrs(
            (
                self.l[(i, v)]
                + self.p[(i, v)]
                - self.l[(j, v)]
                - Q_s[v] * (1 - self.x[(i, j, v)])
                <= 0
                for i, j, v in cart_loc_loc_v
                if i in scooters and j != i
            ),
            "vehicle_capacity_pick_up_less",
        )

        # Add constraints (8):
        self.m.addConstrs(
            (
                self.l[(i, v)]
                + self.p[(i, v)]
                - self.l[(j, v)]
                + Q_s[v] * (1 - self.x[(i, j, v)])
                >= 0
                for i, j, v in cart_loc_loc_v
                if i in scooters and j != i
            ),
            "vehicle_capacity_pick_up_greater",
        )

        # Add constraints (9):
        self.m.addConstrs(
            (
                self.l[(i, v)]
                - self.y[(i, v)]
                - self.l[(j, v)]
                - Q_s[v] * (1 - self.x[(i, j, v)])
                <= 0
                for i, j, v in cart_loc_loc_v
                if i in delivery and j != i
            ),
            "vehicle_capacity_delivery_less",
        )

        # Add constraints (10):
        self.m.addConstrs(
            (
                self.l[(i, v)]
                - self.y[(i, v)]
                - self.l[(j, v)]
                + Q_s[v] * (1 - self.x[(i, j, v)])
                >= 0
                for i, j, v in cart_loc_loc_v
                if i in delivery and j != i
            ),
            "vehicle_capacity_delivery_greater",
        )

        # Add constraints (11):
        self.m.addConstrs(
            (0 <= self.l[(i, v)] for i, v in cart_loc_v if i != depot),
            "vehicle_capacity_cap_noneg",
        )
        self.m.addConstrs(
            (self.l[(i, v)] <= Q_b[v] for i, v in cart_loc_v if i != depot),
            "vehicle_capacity_cap",
        )

        # Add constraints (12):
        self.m.addConstrs(
            (self.l[(0, v)] == 0 for v in service_vehicles), "vehicle_capacity_depot_in"
        )

        # Add constraints (13):
        self.m.addConstrs(
            (
                self.l[(i, v)] - Q_s[v] * (1 - self.x[(0, i, v)]) <= 0
                for i, v in cart_loc_v
                if i != depot
            ),
            "vehicle_capacity_depot_out",
        )

        # Add constraints (14):
        self.m.addConstrs(
            (2 <= self.u[(i, v)] for i, v in cart_loc_v if i != depot), "subtours_1"
        )
        self.m.addConstrs(
            (self.u[(i, v)] <= num_locations for i, v in cart_loc_v if i != depot),
            "subtours_2",
        )

        # Add constraints (15):
        self.m.addConstrs(
            (
                self.u[i, v] - self.u[j, v] + 1
                <= (num_locations - 1) * (1 - self.x[i, j, v])
                for i, j, v in cart_loc_loc_v
                if i != depot
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

        helpers.print_model(self.m)


if __name__ == "__main__":
    m = Model()
    m.setup()
    m.optimize_model()
    m.print_solution()
