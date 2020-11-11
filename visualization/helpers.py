def get_label(instance, i: int):
    """
    Returns the label of a specified index of a model
    :param instance: instance object
    :param i:
    :return:
    """
    if i == 0:
        return "Depot"
    if 0 < i <= instance.model_input.num_scooters:
        return "S"
    else:
        return "D"


def create_node_dict(instance):
    # TODO: should be moved to solution visualization script. this should return a list. Needs documentation
    output = {}
    locations = (
        [instance.depot]
        + list(zip(instance.scooters["lat"], instance.scooters["lon"]))
        + list(zip(instance.delivery_nodes["lat"], instance.delivery_nodes["lon"]))
    )
    for i, index in enumerate(locations):
        output[index] = {"label": get_label(instance, i)}
    return output
