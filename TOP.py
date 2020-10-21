import gurobipy as gp
from gurobipy import GRB
import numpy as np
import helpers
from solution_visualizer import visualize_model_solution

# Create a new model
model = gp.Model("TOP")

# Constants
S = 5  # Number of scooters (scooter 1 and 5 are depot)
V = 1  # Number of service vehicles

# Using random values for R and T
np.random.seed(42)
R = np.random.randint(100, size=(1, S))  # Reward for swapping battery for scooter i
T = np.random.randint(
    10, size=(S + 1, S + 1)
)  # Time needed to travel from scooter i to j
T_max = 100  # Duration of shift
Q_b = 2  # Battery capacity of service vehicle v

"""
Create variables

x_ijv - 1 if, for service vehicle v, visit to scooter i is followed by a visit to scooter j- 0 otherwise
b_iv - 1 if scooter i is visited by service vehicle v- 0 otherwise
u_iv - position of scooter i for service vehicle v route
"""
x, b, u = [{} for i in range(3)]
for i in range(1, S + 1):
    for v in range(1, V + 1):
        b[i, v] = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{v}")
        u[i, v] = model.addVar(vtype=GRB.INTEGER, name=f"u_{i}_{v}")
        for j in range(1, S + 1):
            x[i, j, v] = model.addVar(vtype=GRB.BINARY, name=f"x_({i},{j})_{v}")

# Set objective
model.setObjective(
    gp.quicksum(R[0][i] * b[i, v] for i in range(2, S) for v in range(1, V + 1)),
    GRB.MAXIMIZE,
)

# Add constraints (2): guarantee that each service vehicle starts and ends in at the depot.
model.addConstr(
    gp.quicksum(x[1, j, v] for v in range(1, V + 1) for j in range(2, S + 1)),
    GRB.EQUAL,
    V,
    "must_visit_depot_first",
)
model.addConstr(
    gp.quicksum(x[i, S, v] for v in range(1, V + 1) for i in range(1, S)),
    GRB.EQUAL,
    V,
    "must_visit_depot_end",
)

# Add constraints (3): ensure that every scooter is visited at most once.
for k in range(2, S):
    model.addConstr(
        gp.quicksum(b[k, v] for v in range(1, V + 1)),
        GRB.LESS_EQUAL,
        1,
        f"only_one_visit_pr_scooter_(k={k})",
    )

for v in range(1, V + 1):
    model.addConstr(
        gp.quicksum(b[k, v] for k in range(2, S)),
        GRB.LESS_EQUAL,
        Q_b,
        f"battery_capacity_(k={k})",
    )

# Add constraints (5): guarantee the connectivity of each service vehicle path
for k in range(2, S):
    for v in range(1, V + 1):
        model.addConstr(
            gp.quicksum(x[i, k, v] if i != k else 0 for i in range(1, S)),
            GRB.EQUAL,
            b[k, v],
            f"connectivity_1_(k={k},v={v})",
        )
        model.addConstr(
            gp.quicksum(x[k, j, v] if j != k else 0 for j in range(2, S + 1)),
            GRB.EQUAL,
            b[k, v],
            f"connectivity_2_(k={k},v={v})",
        )

# Add constraints (6): ensure that the length of the paths does not exceed the shift
for v in range(1, V + 1):
    model.addConstr(
        gp.quicksum(T[i, j] * x[i, j, v] for i in range(1, S) for j in range(2, S + 1)),
        GRB.LESS_EQUAL,
        T_max,
        f"time_constraints_(v={v})",
    )

# Add constraints (7-8): prevent subtours
for i in range(2, S + 1):
    for v in range(1, V + 1):
        model.addConstr(2, GRB.LESS_EQUAL, u[i, v], f"subtours_1_(i={i},v={v})")
        model.addConstr(u[i, v], GRB.LESS_EQUAL, S, f"subtours_2_(i={i},v={v})")
        for j in range(2, S + 1):
            if i != j:
                model.addConstr(
                    u[i, v] - u[j, v] + 1,
                    GRB.LESS_EQUAL,
                    (S - 1) * (1 - x[i, j, v]),
                    f"subtours_3_(i,j=({i},{j}),v={v})",
                )

# Optimize model
model.optimize()

visualize_model_solution(model, S)

# helpers.print_model_to_file(model)
