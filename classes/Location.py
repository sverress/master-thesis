class Location:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def get_lat(self):
        return self.lat

    def get_lon(self):
        return self.lon

    def get_location(self):
        return self.lat, self.lon

    def set_location(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon
