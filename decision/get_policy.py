import decision


def get_policy(name="RandomRolloutPolicy"):
    if name == "RandomRolloutPolicy":
        return decision.RandomRolloutPolicy()
    elif name == "SwapAllPolicy":
        return decision.SwapAllPolicy()
    else:
        raise ValueError("No current policy name were given")
