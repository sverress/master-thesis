# Print model
def print_model(model):
    import os
    model.write("model.lp")
    with open("model.lp") as f:
        for line in f.readlines():
            print(line)
    os.remove("model.lp")