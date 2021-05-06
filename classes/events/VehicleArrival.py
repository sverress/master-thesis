from classes.events.Event import Event
import copy


class VehicleArrival(Event):
    def __init__(self, arrival_time: int, vehicle_id: int, visualize=True):
        super().__init__(arrival_time)
        self.visualize = visualize
        self.vehicle_id = vehicle_id

    def perform(self, world, **kwargs) -> None:
        """
        :param world: world object
        """
        try:
            vehicle = [
                vehicle
                for vehicle in world.state.vehicles
                if vehicle.id == self.vehicle_id
            ][0]
        except IndexError:
            raise ValueError(
                "OBS! Something went wrong. The vehicle is not in this state."
            )

        # Remove current vehicle state from tabu list
        world.tabu_list = [
            cluster_id
            for cluster_id in world.tabu_list
            if cluster_id != vehicle.current_location.id
        ]

        if self.visualize:
            # copy state before action for visualization purposes
            state_before_action = copy.deepcopy(world.state)
            vehicle_before_action = copy.deepcopy(vehicle)

        arrival_time = 0

        # if current location is a depot -> refill battery inventory
        if vehicle.is_at_depot():
            batteries_to_swap = min(
                vehicle.current_location.get_available_battery_swaps(world.time),
                vehicle.flat_batteries(),
            )
            arrival_time += vehicle.current_location.swap_battery_inventory(
                world.time, batteries_to_swap
            )
            vehicle.add_battery_inventory(batteries_to_swap)

        # find the best action from the current world state
        action = world.policy.get_best_action(world, vehicle)

        # Add next vehicle location to tabu list
        world.tabu_list.append(action.next_location)

        if self.visualize:
            # visualize vehicle route
            world.state.visualize_vehicle_routes(
                self.vehicle_id,
                vehicle.current_location.id,
                action.next_location,
                world.tabu_list,
                world.policy.__str__(),
            )

        # clear world flow counter dictionary
        world.clear_flow_dict()

        # Record current location of vehicle to compute action time
        arrival_cluster_id = vehicle.current_location.id

        # perform the best action on the state and send vehicle to new location
        reward = world.state.do_action(action, vehicle, world.time)

        world.add_reward(reward, arrival_cluster_id, discount=True)

        if self.visualize:
            # visualize action performed by vehicle
            state_before_action.visualize_action(
                vehicle_before_action,
                world.state,
                vehicle,
                action,
                False,
                world.policy.__str__(),
            )

        # set time of world to this event's time
        super(VehicleArrival, self).perform(world, **kwargs)

        # Compute the arrival time for the Vehicle arrival event created by the action
        arrival_time += self.time + action.get_action_time(
            world.state.get_distance(arrival_cluster_id, action.next_location)
        )

        # Add a new Vehicle Arrival event for the next cluster arrival to the world stack
        world.add_event(VehicleArrival(arrival_time, vehicle.id, self.visualize))
