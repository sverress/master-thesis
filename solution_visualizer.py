import gurobipy as gp
import matplotlib.pyplot as plt
import networkx as nx


def visualize_model_solution(model: gp.Model, number_of_nodes: int):
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(1, number_of_nodes + 1)])
    for var in model.getVars():
        # Get arcs
        if var.varName.startswith("x") and var.x > 0:
            print(f"{var.varName} {var.x}")
            graph.add_edge(
                *tuple(
                    [
                        int("".join([c for c in string if c not in "()"]))
                        for string in var.varName.split("_")[1].split(",")
                    ]
                )
            )
    nx.draw(graph, with_labels=True, font_weight="bold")
    plt.show()
