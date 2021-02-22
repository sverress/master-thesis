from globals import GEOSPATIAL_BOUND_NEW


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
        lat_min, lat_max, lon_min, lon_max = GEOSPATIAL_BOUND_NEW
        if lat < lat_min:
            self.lat = lat_min
        elif lat > lat_max:
            self.lat = lat_max
        else:
            self.lat = lat

        if lon < lon_min:
            self.lon = lon_min
        elif lon > lon_max:
            self.lon = lon_max
        else:
            self.lon = lon
