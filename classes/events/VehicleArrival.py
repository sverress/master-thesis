from classes.events import Event


class VehicleArrival(Event):
    def __init__(self, arrival_time: int, arrival_cluster_id: int):
        super().__init__(arrival_time)
        self.arrival_cluster_id = arrival_cluster_id

    def perform(self, world) -> None:
        """
        :param world: world object
        """
        # get the cluster object that the vehicle has arrived to
        arrival_cluster = world.state.get_cluster_by_id(self.arrival_cluster_id)

        # set the arrival cluster as current cluster in state
        world.state.current_cluster = arrival_cluster

        # TODO when implemented in decision: decision.get_best_action(state) -> action
        action = None

        # perform the best action on the state
        reward = world.state.do_action(action)

        # add the reward from the action to a reward list for a posterior analysis
        world.add_reward(reward)

        # set time of world to this event's time
        super(VehicleArrival, self).perform(world)
