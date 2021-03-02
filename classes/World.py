import clustering.scripts as clustering_scripts


class World:
    def __init__(self, shift_duration: int, sample_size=100, number_of_clusters=200):
        self.shift_duration = shift_duration
        self.state = clustering_scripts.get_initial_state(
            sample_size=sample_size, number_of_clusters=number_of_clusters
        )
        self.stack = []
        self.time = 0

    def run(self):
        while self.time < self.shift_duration:
            self.stack.pop().perform(self)
