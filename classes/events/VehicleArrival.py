from classes import Event
import copy


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

        # find the best action from the current world state
        action = world.policy.get_best_action(world)

        # visualize cluster flows since last vehicle arrival
        world.state.visualize_flow(world.get_cluster_flow(), action.next_cluster)

        # clear world flow counter dictionary
        world.clear_flow_dict()

        # copy state before action for visualization purposes
        state_before_action = copy.deepcopy(world.state)

        # perform the best action on the state
        reward = world.state.do_action(action)

        # visualize action performed by vehicle
        state_before_action.visualize_action(world.state, action)

        # add the reward from the action to a reward list for a posterior analysis
        world.add_reward(reward)

        # set time of world to this event's time
        super(VehicleArrival, self).perform(world)

        # Compute the arrival time for the Vehicle arrival event created by the action
        arrival_time = self.time + action.get_action_time(
            world.state.get_distance_id(self.arrival_cluster_id, action.next_cluster)
        )
        # Add a new Vehicle Arrival event for the next cluster arrival to the world stack
        world.add_event(VehicleArrival(arrival_time, action.next_cluster))
