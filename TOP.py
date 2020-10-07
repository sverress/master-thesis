import gurobipy as gp
from gurobipy import GRB
import numpy as np
import helpers

# Create a new model
model = gp.Model("TOP")

# Constants and sets
S = 5  # Number of scooters (scooter 1 and 5 are depot)
V = 2  # Number of service vehicles

# Using random values for R and T
np.random.seed(42)
R = np.random.randint(100, size=(1, S))  # Reward for swapping battery for scooter i
T = np.random.randint(10, size=(S, S))  # Time needed to travel from scooter i to j
T_max = 10  # Duration of shift

"""
Create variables

x_ijv - 1 if, for service vehicle v, visit to scooter i is followed by a visit to scooter j- 0 otherwise
y_iv - 1 if scooter i is visited by service vehicle v- 0 otherwise
u_iv - position of scooter i for service vehicle v route
"""
x, y, u = [{} for i in range(3)]
for i in range(1, S+1):
    for v in range(1, V + 1):
        y[i, v] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{v}")
        u[i, v] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{v}")
        for j in range(1, S + 1):
            x[i, j, v] = model.addVar(vtype=GRB.BINARY, name=f"x_({i},{j})_{v}")

# Set objective
model.setObjective(gp.quicksum(R[0][i] * y[i, v] for i in range(2, S) for v in range(1, V+1)), GRB.MAXIMIZE)

# Add constraint (2): guarantee that each service vehicle starts and ends in at the depot.
model.addConstr(gp.quicksum(x[1, j, v] for v in range(1, V+1) for j in range(2, S+1)), GRB.EQUAL, V, "c_2_1")
model.addConstr(gp.quicksum(x[i, S, v] for v in range(1, V+1) for i in range(1, S)), GRB.EQUAL, V, "c_2_2")

# Add constraint (3): ensure that every scooter is visited at most once.
for k in range(2, S):
    model.addConstr(gp.quicksum(y[k, v] for v in range(1, V+1)), GRB.LESS_EQUAL, 1, f"c_3_{k}")

# Add constraint (4): guarantee the connectivity of each service vehicle path
for k in range(2, S):
    for v in range(1, V+1):
        model.addConstr(gp.quicksum(x[i, k, v] for i in range(1, S)), GRB.EQUAL, y[k, v], f"c_4_1_{k},{v}")
        model.addConstr(gp.quicksum(x[k, j, v] for j in range(2, S+1)), GRB.EQUAL, y[k, v], f"c_4_2_{k},{v}")

# Optimize model
model.optimize()

# Print solution
for v in model.getVars():
    if v.x > 0:
        print(f'{v.varName} {int(v.x)}')

print(f'Obj: {model.objVal}')

helpers.print_model(model)
