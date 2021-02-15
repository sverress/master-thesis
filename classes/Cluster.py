from classes.Scooter import Scooter


class Cluster:
    def __init__(self, scooters: [Scooter]):
        # sorting scooters after battery percent
        self.scooters = sorted(scooters, key=lambda s: s.battery, reverse=True)
        self.current_state = self.__compute_current_state()
        self.ideal_state = 2
        self.trip_intensity_per_iteration = 2
        self.center = (scooters[0].lat, scooters[0].lon)

    def __compute_current_state(self):
        return sum(map(lambda scooter: scooter.battery, self.scooters))

    def dist(self, cluster):
        return 5

    def prob_stay(self):
        return 0.5

    def prob_leave(self, cluster):
        return 0.5 / 7  # 7 is number of clusters

    def number_of_possible_pickups(self):
        if self.number_of_scooters() > self.ideal_state:
            return 0
        else:
            return self.number_of_scooters()

    def number_of_scooters(self):
        return len(self.scooters)

    def total_battery_percent(self):
        return sum([s.battery for s in self.scooters])

    def add_scooter(self, scooter: Scooter):
        self.scooters.append(scooter)
        self.scooters.sort(key=lambda s: s.battery, reverse=True)

    def to_string(self):
        string = ""
        for s in self.scooters:
            string += f"ID: {s.id}  Battery {round(s.battery,1)} | "
        return string
