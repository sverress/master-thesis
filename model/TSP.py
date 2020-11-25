import gurobipy as gp
from gurobipy import GRB
from itertools import product


def TSP(number_of_nodes, time_matrix):
    m = gp.Model("TSP")
    nodes = range(number_of_nodes)
    cart_nodes = list(product(nodes, repeat=2))
    x = m.addVars(cart_nodes, vtype=GRB.BINARY, name="x")
    u = m.addVars(nodes, vtype=GRB.CONTINUOUS, name="u")

    m.addConstrs((gp.quicksum(x[(i, j)] for i in nodes if i != j) == 1) for j in nodes)
    m.addConstrs((gp.quicksum(x[(i, j)] for j in nodes if i != j) == 1) for i in nodes)
    m.addConstrs(
        (u[i] - u[j] + number_of_nodes * x[(i, j)] <= number_of_nodes - 1)
        for i, j in cart_nodes
        if i != j and i > 0
    )
    m.addConstrs((1 <= u[i]) for i in range(1, number_of_nodes))
    m.addConstrs((u[i] <= number_of_nodes - 1) for i in range(1, number_of_nodes))
    m.setObjective(
        gp.quicksum(time_matrix[(i, j)] * x[(i, j)] for i, j in cart_nodes if i != j),
        GRB.MINIMIZE,
    )
    m.Params.TimeLimit = 2
    m.Params.OutputFlag = 0
    m.optimize()
    return m.objVal
