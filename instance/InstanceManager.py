import pandas as pd
import random
import math

from instance.helpers import *
from instance.Instance import Instance
from model.StandardModel import StandardModel
from model.AlternativeModel import AlternativeModel
from model.BaseModelInput import BaseModelInput


class InstanceManager:
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
        self, number_of_sections: int, number_of_scooters_per_section: int, **kwargs
    ):
        """
        Creates necessary data structures to create a new ModelInput object.

        TODO: This methods input parameters are only related to zones and number of nodes.
        TODO: Other parameters should be added optionally
        :param number_of_sections: number of sections at each x and y axis. ex. 3 gives 9 zones
        :param number_of_scooters_per_section: this is the number of scooters per zone and is also considered the optimal
        state at the time of writing
        :return: Instance object
        """
        number_of_scooters = number_of_scooters_per_section * number_of_sections ** 2
        number_of_vehicles = kwargs.get(
            "number_of_vehicles", math.ceil(number_of_scooters / 10)
        )

        filtered_scooters = self.filter_data_lat_lon(self._data, self._bound)
        scooters = filtered_scooters.sample(
            number_of_scooters, random_state=self._random_state
        )[["lat", "lon", "battery"]]

        scooters["zone"] = -1

        delivery_nodes = self.create_all_delivery_nodes(
            number_of_sections, scooters, number_of_scooters_per_section
        )
        # Giving dataframes same index as they will have in mathematical model
        scooters.index = range(1, len(scooters) + 1)
        delivery_nodes.index = range(
            len(scooters) + 1, len(scooters) + len(delivery_nodes) + 1
        )
        # Creating depot node in the middle of bound
        depot = get_center_of_bound(self._bound)
        number_of_car_service_vehicles = math.ceil(number_of_vehicles / 2)
        number_of_bike_service_vehicles = math.floor(number_of_vehicles / 2)
        service_vehicles = {
            "car": (number_of_car_service_vehicles, 5, 10),
            "bike": (number_of_bike_service_vehicles, 0, 3),
        }

        is_percent_t_max = kwargs.get("T_max_is_percentage", True)
        if is_percent_t_max or kwargs.get("T_max") is None:
            # Combine all locations in one dataframe
            all_locations = pd.concat(
                (
                    pd.DataFrame([list(depot)], columns=["lat", "lon"]),
                    scooters[["lat", "lon"]],
                    delivery_nodes[["lat", "lon"]],
                )
            )
            sum_of_travel_time = 0.0
            number_of_iterations = 5
            for k in range(number_of_iterations):
                dataset = all_locations.sample(frac=1)
                previous_cords = dataset.iloc[0]
                for i in range(1, len(all_locations)):
                    current_cords = dataset.iloc[i]
                    sum_of_travel_time += BaseModelInput.compute_distance(
                        previous_cords["lat"],
                        previous_cords["lon"],
                        current_cords["lat"],
                        current_cords["lon"],
                    )
                    previous_cords = current_cords
            t_max = (
                (sum_of_travel_time / (number_of_iterations * number_of_vehicles))
                * 0.5  # Tuned parameter
                * kwargs.get("T_max", 0.70)
            )
        else:
            t_max = kwargs.get("T_max")
        return Instance(
            scooters,
            delivery_nodes,
            depot,
            service_vehicles,
            number_of_sections,
            t_max,
            kwargs.get("time_limit", 10),
            self._bound,
            AlternativeModel
            if kwargs.get("model_type", "standard") == "alternative"
            else StandardModel,
        )

    def create_multiple_instances(self):
        """
        Generates multiple instances and stores them to the instances dict.
        Instance parameters are loaded from json file
        """
        instances_parameters = load_test_parameters_from_json()
        for parameters in instances_parameters:
            # TODO: Should be possible to choose what model type to use in json config
            instance_parameters = *parameters, StandardModel
            self.instances[parameters] = self.create_test_instance(*instance_parameters)

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
    def filter_data_lat_lon(
        geospatial_data: pd.DataFrame, coordinates: tuple, only_expression=False
    ):
        """
        :param geospatial_data: dataframe with lat lon columns
        :param coordinates: tuple with a size of four to define area to include locations from
        :param only_expression: if only expression should be returned
        :return: new dataframe containing locations within the given coordinates
        """
        lat_min, lat_max, lon_min, lon_max = coordinates
        expression = (
            (lon_min <= geospatial_data["lon"])
            & (geospatial_data["lon"] <= lon_max)
            & (lat_min <= geospatial_data["lat"])
            & (geospatial_data["lat"] <= lat_max)
        )
        return expression if only_expression else geospatial_data.loc[expression]

    @staticmethod
    def create_delivery_nodes(
        scooters, coordinates, optimal_number, additional_cols: list
    ):
        """
        Generates delivery nodes for a zone. The number of generated delivery nodes is the difference between the
        optimal state and the number of scooters in the area. The nodes are now generated randomly within the zone.
        :param scooters: Dataframe containing the locations of all scooters
        :param coordinates: tuple defining the bounds of the zone
        :param optimal_number: the optimal number of scooters in this zone
        :return: list of tuples with lat lon of the generated delivery locations
        """
        distance = 0.002
        filtered_df = InstanceManager.filter_data_lat_lon(scooters, coordinates)
        center_lat, center_lon = get_center_of_bound(coordinates)

        def get_deviation():
            return random.uniform(-distance, distance)

        return [
            (
                center_lat + get_deviation(),
                center_lon + get_deviation(),
                *additional_cols,
            )
            for i in range(optimal_number - len(filtered_df))
        ]

    def get_instance(self, instance_id: tuple):
        """
        Fetch the instance from the instance id given and use the visualize_test_data function to show this instance
        :param instance_id: tuple(number of sections, number of scooters pr section)
        :return: 
        """
        return self.instances[instance_id]

    def create_all_delivery_nodes(
        self, num_of_sections, scooters, num_of_scooters_per_section
    ):
        sections, sections_coordinates = create_sections(num_of_sections, self._bound)
        delivery_nodes = []
        for i, bound_coordinates in enumerate(sections_coordinates):
            scooters.loc[
                self.filter_data_lat_lon(scooters, bound_coordinates, True), "zone"
            ] = i  # Set zone id
            delivery_nodes = delivery_nodes + self.create_delivery_nodes(
                scooters, bound_coordinates, num_of_scooters_per_section, [i]
            )
        return pd.DataFrame(delivery_nodes, columns=["lat", "lon", "zone"])
