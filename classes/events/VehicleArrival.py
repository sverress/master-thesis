from classes.events import Event


class VehicleArrival(Event):
    def __init__(self, arrival_time: int):
        super().__init__(arrival_time)

    def perform(self, world) -> None:
        """
        :param world: world object
        """

        # TODO when implemented in decision: decision.get_best_action(state) -> action
        action = None

        # perform the best action on the state
        reward = world.state.do_action(action)

        # add the reward from the action to a reward list for a posterior analysis
        world.add_reward(reward)

        # set time of world to this event's time
        world.time = self.time
