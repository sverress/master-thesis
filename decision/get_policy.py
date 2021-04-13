import decision
import classes
from globals import NUMBER_OF_CLUSTERS, EPSILON


def get_policy(
    name="RandomRolloutPolicy",
    value_function="ValueFunction",
    number_of_clusters=NUMBER_OF_CLUSTERS,
    epsilon=EPSILON,
):
    if name == "RandomRolloutPolicy":
        return decision.RandomRolloutPolicy()
    elif name == "SwapAllPolicy":
        return decision.SwapAllPolicy()
    elif name == "TD0Policy":
        return decision.TD0Policy(
            classes.ValueFunction.get_value_function(
                value_function, number_of_clusters
            ),
            epsilon,
        )
    elif name == "RandomActionPolicy":
        return decision.RandomActionPolicy()
    else:
        raise ValueError("No current policy name were given")
