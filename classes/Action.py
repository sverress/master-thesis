from globals import MINUTES_IN_HOUR, VEHICLE_SPEED


class Action:
    def __init__(
        self,
        battery_swaps: [int],
        pick_ups: [int],
        delivery_scooters: [int],
        next_location: int,
    ):
        """
        Object to represent an action
        :param battery_swaps: ids of scooters to swap batteries on
        :param pick_ups: ids of scooters to pick up
        :param delivery_scooters: ids of scooters to deliver
        :param next_location: id of next location to visit
        """
        self.battery_swaps = battery_swaps
        self.pick_ups = pick_ups
        self.delivery_scooters = delivery_scooters
        self.next_location = next_location

    def get_action_time(self, distance):
        """
        Get the time consumed from performing an action (travel from cluster 1 to 2) in a given state.
        Can add time for performing actions on scooters as well.
        :param distance: distance in km from current cluster to next cluster
        :return: Total time to perform action in minutes
        """
        operation_duration = (
            len(self.battery_swaps) + len(self.pick_ups) + len(self.delivery_scooters)
        ) * 2
        travel_duration = round((distance / VEHICLE_SPEED) * MINUTES_IN_HOUR)
        return operation_duration + travel_duration

    def __repr__(self):
        return (
            f"<Action - ({len(self.battery_swaps)} bat. swaps, {len(self.pick_ups)} pickups,"
            f" {len(self.delivery_scooters)} deliveries), next: {self.next_location} >"
        )
