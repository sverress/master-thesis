from math import sqrt


def print_model(model, delete_file=True):
    import os

    model.write("model.lp")
    with open("model.lp") as f:
        for line in f.readlines():
            print(line)
    if delete_file:
        os.remove("model.lp")


def compute_distance(loc1, loc2):
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return sqrt(dx * dx + dy * dy)
