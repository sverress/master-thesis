import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math


class TestInstanceManager:
    def __init__(self):
        self._data_file_path = "test_data/bigquery-results.csv"
        self._data = self.get_data()
        self._bound = (
            59.9112,
            59.9438,
            10.7027,
            10.7772,
        )  # (lat_min, lat_max, lon_min, lon_max) Area in Oslo
        self._epsilon = 0.000001  # Just a small number
        self._random_state = 1
        random.seed(self._random_state)

    def create_test_instance(
        self, num_of_sections: int, num_of_scooters_per_section: int,
    ):
        num_of_scooters = num_of_scooters_per_section * num_of_sections ** 2
        filtered_scooters = self.filter_data_lat_lon(self._data, self._bound)
        scooters = filtered_scooters.sample(
            num_of_scooters, random_state=self._random_state
        )[["lat", "lon", "battery"]]

        sections, sections_coordinates = self.create_sections(num_of_sections)
        delivery_nodes = []
        for bound_coordinates in sections_coordinates:
            delivery_nodes = delivery_nodes + self.create_delivery_nodes(
                scooters, bound_coordinates, num_of_scooters_per_section
            )
        delivery_nodes = pd.DataFrame(delivery_nodes, columns=["lat", "lon"])
        lat_min, lat_max, lon_min, lon_max = self._bound
        depot = (lat_max - lat_min) / 2, (lon_max - lon_min) / 2
        num_of_car_service_vehicles = math.ceil(num_of_scooters / 10)
        service_vehicles = {"car": (num_of_car_service_vehicles, 10, 30)}
        if num_of_scooters % 10 > 5:
            service_vehicles["bike"] = (1, 0, 5)
        return scooters, delivery_nodes, depot, service_vehicles

    def set_random_state(self, new_state: int):
        self._random_state = new_state
        random.seed(self._random_state)

    def get_data(self, separator=";"):
        df = pd.read_csv(self._data_file_path, sep=separator)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    @staticmethod
    def filter_data_lat_lon(df: pd.DataFrame, coordinates: tuple):
        lat_min, lat_max, lon_min, lon_max = coordinates
        return df[
            (lon_min <= df["lon"])
            & (df["lon"] <= lon_max)
            & (lat_min <= df["lat"])
            & (df["lat"] <= lat_max)
        ]

    def create_sections(self, number_of_sections: int):
        lat_min, lat_max, lon_min, lon_max = self._bound
        lon_length = lon_max - lon_min
        lat_length = lat_max - lat_min
        section_limits = pd.DataFrame()
        section_limits["lon"] = np.arange(
            lon_min, lon_max + self._epsilon, lon_length / number_of_sections
        )
        section_limits["lat"] = np.arange(
            lat_min, lat_max + self._epsilon, lat_length / number_of_sections
        )
        section_coordinates = []
        for j in range(1, number_of_sections + 1):
            for i in range(1, number_of_sections + 1):
                section_coordinates.append(
                    (
                        section_limits["lat"][j - 1],
                        section_limits["lat"][j],
                        section_limits["lon"][i - 1],
                        section_limits["lon"][i],
                    )
                )
        return section_limits, section_coordinates

    @staticmethod
    def create_delivery_nodes(df, coordinates, optimal_number):
        lat_min, lat_max, lon_min, lon_max = coordinates
        filtered_df = TestInstanceManager.filter_data_lat_lon(df, coordinates)
        return [
            (random.uniform(lat_min, lat_max), random.uniform(lon_min, lon_max))
            for i in range(optimal_number - len(filtered_df))
        ]

    def visualize_test_data(
        self, scooters: pd.DataFrame, delivery_nodes: pd.DataFrame, sections=None,
    ):
        lat_min, lat_max, lon_min, lon_max = self._bound
        fig, ax = plt.subplots()

        ax.scatter(scooters["lon"], scooters["lat"], zorder=1, alpha=0.4, s=10)

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        if sections is not None:
            ax.set_xticks(sections["lon"])
            ax.set_yticks(sections["lat"])
            ax.grid()

        ax.scatter(
            delivery_nodes["lon"],
            delivery_nodes["lat"],
            zorder=1,
            alpha=0.4,
            s=10,
            c="r",
        )

        oslo = plt.imread("test_data/oslo.png")
        ax.imshow(
            oslo,
            zorder=0,
            extent=(lon_min, lon_max, lat_min, lat_max),
            aspect="equal",
            alpha=0.4,
        )

        plt.show()


if __name__ == "__main__":
    manager = TestInstanceManager()
    NUM_OF_SECTIONS = 2
    SCOOTERS_IN_A_ZONE = 3
    scooters, delivery_nodes, depot, service_vehicles = manager.create_test_instance(
        NUM_OF_SECTIONS, SCOOTERS_IN_A_ZONE
    )
    sections, sections_coordinates = manager.create_sections(NUM_OF_SECTIONS)
    manager.visualize_test_data(scooters, delivery_nodes, sections)
