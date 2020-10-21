# Print model
def print_model_to_file(model, delete_file=True):
    import os

    model.write("model.lp")
    with open("model.lp") as f:
        for line in f.readlines():
            print(line)
    if delete_file:
        os.remove("model.lp")
