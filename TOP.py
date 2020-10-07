import gurobipy as gp
from gurobipy import GRB
import numpy as np

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
        y[i, v] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}{v}")
        u[i, v] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}{v}")
        for j in range(1, S + 1):
            x[i, j, v] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}{j}{v}")
# Set objective
model.setObjective(gp.quicksum(R[0][i] * y[i, v] for i in range(2, S) for v in range(1, V+1)), GRB.MAXIMIZE)

# Add constraint (3): ensure that every scooter is visited at most once.
for k in range(2, S):
    model.addConstr(gp.quicksum(y[k, v] for v in range(1, V+1)), GRB.LESS_EQUAL, 1, f"c_3_{k}")

# Optimize model
model.optimize()

for v in model.getVars():
    if v.varName.startswith("y"):
        print('%s %g' % (v.varName, v.x))

print('Obj: %g' % model.objVal)