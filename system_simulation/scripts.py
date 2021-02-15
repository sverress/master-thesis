from classes.State import State


def system_simulate(state: State):
    return state


# What to do:

# Loope over alle clustere

## ANTAR HER AT DET IKKE KOMMER INN NYE SYKLER (derfor må man ikke flytte sykler til clustre som ikke har gjennomgått prosessen)

# Utfør en poisson process for "ankomst av turer" i et cluster
# For alle turer i tidsintervallet -> finn hvor den skal dra og plukk en sykkel
#
