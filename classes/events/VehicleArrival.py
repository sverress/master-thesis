import classes
from classes import Event
from globals import BATTERY_INVENTORY
import copy


class VehicleArrival(Event):
    def __init__(self, arrival_time: int, arrival_location_id: int, visualize=True):
        super().__init__(arrival_time)
        self.arrival_location_id = arrival_location_id
        self.visualize = visualize

    def perform(self, world) -> None:
        """
            :param world: world object
            """
        # copy state before action for visualization purposes
        state_before_action = copy.deepcopy(world.state)

        arrival_time = 0

        # get the cluster object that the vehicle has arrived to
        arrival_cluster = world.state.get_location_by_id(self.arrival_location_id)

        # set the arrival cluster as current cluster in state
        world.state.current_location = arrival_cluster

        # if current location is a depot -> refill battery inventory
        if isinstance(world.state.current_location, classes.Depot):
            batteries_to_swap = min(
                world.state.current_location.get_available_battery_swaps(world.time),
                BATTERY_INVENTORY - world.state.vehicle.battery_inventory,
            )
            arrival_time += world.state.current_location.swap_battery_inventory(
                world.time, batteries_to_swap
            )
            world.state.vehicle.add_battery_inventory(batteries_to_swap)

        # find the best action from the current world state
        action = world.policy.get_best_action(world)

        # add the cluster id for the cluster the vehicle arrives at to the vehicles trip
        world.state.vehicle.add_cluster_id_to_route(world.state.current_location.id)

        if self.visualize:
            # visualize vehicle route
            world.state.visualize_vehicle_route(
                world.state.vehicle.get_route(), action.next_location,
            )

            # visualize scooters currently out on a trip
            world.state.visualize_current_trips(world.get_scooters_on_trip())

        # clear world flow counter dictionary
        world.clear_flow_dict()

        # perform the best action on the state
        reward = world.state.do_action(action)

        if self.visualize:
            # visualize action performed by vehicle
            state_before_action.visualize_action(world.state, action)

        # add the reward from the action to a reward list for a posterior analysis
        world.add_reward(reward)

        # set time of world to this event's time
        super(VehicleArrival, self).perform(world)

        # Compute the arrival time for the Vehicle arrival event created by the action
        arrival_time += self.time + action.get_action_time(
            world.state.get_distance_id(self.arrival_location_id, action.next_location)
        )

        # Add a new Vehicle Arrival event for the next cluster arrival to the world stack
        world.add_event(
            VehicleArrival(arrival_time, action.next_location, self.visualize)
        )
