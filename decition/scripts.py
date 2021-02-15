from clustering.scripts import get_initial_state
from classes.Action import Action
from classes.State import State
from scenario_simulation.scripts import estimate_reward
from system_simulation.scripts import system_simulate


def get_state(action: Action, state: State):
    state.current = action.next_cluster
    # Perform pickups
    for pick_up in action.pick_ups:
        state.vehicle.pick_up(pick_up)
    # Perform battery change
    state.vehicle.change_batteries(len(action.battery_swaps))
    return state


def run():
    # Get data from database
    state = get_initial_state()

    # Find all possible actions
    action = Action([], [], [], state.clusters[0])
    new_state = get_state(action, state)

    # Estimate value of making this action
    reward = estimate_reward(new_state)

    # Choose an action

    # System simulation
    state = system_simulate(new_state)


run()
