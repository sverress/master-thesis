import decision
from globals import EPSILON


def get_policy(
    name="RandomRolloutPolicy", value_function=None, state=None, epsilon=EPSILON,
):
    if name == "RandomRolloutPolicy":
        return decision.RandomRolloutPolicy()
    elif name == "SwapAllPolicy":
        return decision.SwapAllPolicy()
    elif name == "TD0Policy":
        return decision.TD0Policy(value_function(state), epsilon)
    elif name == "RandomActionPolicy":
        return decision.RandomActionPolicy()
    else:
        raise ValueError("No current policy name were given")
