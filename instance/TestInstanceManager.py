import pandas as pd
from instance.helpers import create_sections
import random
import math

from instance.Instance import Instance


class TestInstanceManager:
    def __init__(self):
        """
        Class for creating multiple test instances from a given EnTur Dataset snapshot.
        The class contains methods for fetching data, cleaning it and creating Instance instances to later be runned.
        This class also contains methods for visualizing the incoming data.
        """
        self._data_file_path = "test_data/bigquery-results.csv"
        self._data = self.get_data()
        self._bound = (
            59.9112,
            59.9438,
            10.7027,
            10.7772,
        )  # (lat_min, lat_max, lon_min, lon_max) Area in Oslo
        self._random_state = 1
        random.seed(self._random_state)
        self.instances = (
            {}
        )  # Instances indexed by (num_of_sections, num_of_scooters_per_section)

    def create_test_instance(
        self, num_of_sections: int, num_of_scooters_per_section: int,
    ):
        """
        Creates necessary data structures to create a new ModelInput object.

        TODO: This methods input parameters are only related to zones and number of nodes.
        TODO: Other parameters should be added optionally
        :param num_of_sections: number of sections at each x and y axis. ex. 3 gives 9 zones
        :param num_of_scooters_per_section: this is the number of scooters per zone and is also considered the optimal
        state at the time of writing
        :return: Input arguments to ModelInput class
        """
        num_of_scooters = num_of_scooters_per_section * num_of_sections ** 2
        filtered_scooters = self.filter_data_lat_lon(self._data, self._bound)
        scooters = filtered_scooters.sample(
            num_of_scooters, random_state=self._random_state
        )[["lat", "lon", "battery"]]

        sections, sections_coordinates = create_sections(num_of_sections, self._bound)
        delivery_nodes = []
        for bound_coordinates in sections_coordinates:
            delivery_nodes = delivery_nodes + self.create_delivery_nodes(
                scooters, bound_coordinates, num_of_scooters_per_section
            )
        delivery_nodes = pd.DataFrame(delivery_nodes, columns=["lat", "lon"])
        lat_min, lat_max, lon_min, lon_max = self._bound
        depot = lat_min + (lat_max - lat_min) / 2, lon_min + (lon_max - lon_min) / 2
        num_of_car_service_vehicles = math.ceil(num_of_scooters / 10)
        service_vehicles = {
            "car": (num_of_car_service_vehicles, 10, 30),
            "bike": (0, 0, 5,),
            # Zero bikes at init to fix key value error in visualization
        }
        if num_of_scooters % 10 > 5:
            service_vehicles["bike"] = (1, 0, 5)
        return Instance(
            scooters,
            delivery_nodes,
            depot,
            service_vehicles,
            num_of_sections,
            self._bound,
        )

    def create_multiple_instances(self, instances_parameters: list):
        """
        Generates multiple instances and stores them to the instances dict
        :param instances_parameters: list of instance parameters.
        ex: [(2,3)] -> add one instance with 2 sections and 3 scooters pr section
        """
        for parameters in instances_parameters:
            num_of_sections, num_of_scooters_per_section = parameters
            self.instances[parameters] = self.create_test_instance(
                num_of_sections, num_of_scooters_per_section
            )

    def set_random_state(self, new_state: int):
        """
        Function to change the random state (pandas thing) of the object. The random state is also used as the seed
        :param new_state: integer number of the new random state
        """
        self._random_state = new_state
        random.seed(self._random_state)

    def get_data(self, separator=";"):
        """
        Fetches data from csv file and adds datetime type to timestamp column.
        This would be the place for additional data cleaning
        :param separator:
        :return:
        """
        df = pd.read_csv(self._data_file_path, sep=separator)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    @staticmethod
    def filter_data_lat_lon(geospatial_data: pd.DataFrame, coordinates: tuple):
        """
        :param geospatial_data: dataframe with lat lon columns
        :param coordinates: tuple with a size of four to define area to include locations from
        :return: new dataframe containing locations within the given coordinates
        """
        lat_min, lat_max, lon_min, lon_max = coordinates
        return geospatial_data[
            (lon_min <= geospatial_data["lon"])
            & (geospatial_data["lon"] <= lon_max)
            & (lat_min <= geospatial_data["lat"])
            & (geospatial_data["lat"] <= lat_max)
        ]

    @staticmethod
    def create_delivery_nodes(scooters, coordinates, optimal_number):
        """
        Generates delivery nodes for a zone. The number of generated delivery nodes is the difference between the
        optimal state and the number of scooters in the area. The nodes are now generated randomly within the zone.
        :param scooters: Dataframe containing the locations of all scooters
        :param coordinates: tuple defining the bounds of the zone
        :param optimal_number: the optimal number of scooters in this zone
        :return: list of tuples with lat lon of the generated delivery locations
        """
        lat_min, lat_max, lon_min, lon_max = coordinates
        filtered_df = TestInstanceManager.filter_data_lat_lon(scooters, coordinates)
        return [
            (random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max))
            for i in range(optimal_number - len(filtered_df))
        ]

    def get_instance(self, instance_id: tuple):
        """
        Fetch the instance from the instance id given and use the visualize_test_data function to show this instance
        :param instance_id: tuple(number of sections, number of scooters pr section)
        :return: 
        """
        return self.instances[instance_id]


if __name__ == "__main__":
    manager = TestInstanceManager()
    parameter_list = [(2, 4), (2, 5), (3, 3)]
    manager.create_multiple_instances(parameter_list)
    instance = manager.get_instance((2, 4))
    instance.run()
    instance.model.print_solution()
    instance.visualize_solution()
