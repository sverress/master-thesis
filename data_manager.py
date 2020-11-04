import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

OSLO_LAT_LON = (10.7027, 10.7772, 59.9112, 59.9438)
random.seed(42)


def get_raw_data(sections: int, scooters_per_section: int):
    df = pd.read_csv("bigquery-results.csv", sep=";")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Filter out scooters not in Oslo lat long area
    return filter_data_lat_lon(df, OSLO_LAT_LON).head(
        scooters_per_section * sections ** 2
    )


def filter_data_lat_lon(df, coordinates):
    return df[
        (coordinates[0] <= df["lon"])
        & (df["lon"] <= coordinates[1])
        & (coordinates[2] <= df["lat"])
        & (df["lat"] <= coordinates[3])
    ]


def visualize_test_data(scooters: pd.DataFrame, number_of_sections: int):
    fig, ax = plt.subplots()

    ax.scatter(scooters["lon"], scooters["lat"], zorder=1, alpha=0.4, s=10)

    ax.set_xlim(OSLO_LAT_LON[0], OSLO_LAT_LON[1])
    ax.set_ylim(OSLO_LAT_LON[2], OSLO_LAT_LON[3])

    sections, sections_coordinates = create_sections(number_of_sections)
    ax.set_xticks(sections["lon"])
    ax.set_yticks(sections["lat"])
    ax.grid()

    delivery_nodes = []
    for coordinates in sections_coordinates:
        delivery_nodes = delivery_nodes + create_delivery_nodes(
            scooters, coordinates, 20
        )
    delivery_nodes = pd.DataFrame(delivery_nodes, columns=["lon", "lat"])
    ax.scatter(
        delivery_nodes["lon"], delivery_nodes["lat"], zorder=1, alpha=0.4, s=10, c="r"
    )

    oslo = plt.imread("oslo.png")
    ax.imshow(oslo, zorder=0, extent=OSLO_LAT_LON, aspect="equal", alpha=0.4)

    plt.show()


def create_delivery_nodes(df, coordinates, optimal_number):
    filtered_df = filter_data_lat_lon(df, coordinates)
    return [
        (
            random.uniform(coordinates[0], coordinates[1]),
            random.uniform(coordinates[2], coordinates[3]),
        )
        for i in range(optimal_number - len(filtered_df))
    ]


def create_sections(number_of_sections):
    lon_length = OSLO_LAT_LON[1] - OSLO_LAT_LON[0]
    lat_length = OSLO_LAT_LON[3] - OSLO_LAT_LON[2]
    df = pd.DataFrame()
    df["lon"] = np.arange(
        OSLO_LAT_LON[0], OSLO_LAT_LON[1] + 0.000001, lon_length / number_of_sections
    )
    df["lat"] = np.arange(
        OSLO_LAT_LON[2], OSLO_LAT_LON[3] + 0.000001, lat_length / number_of_sections
    )
    sections = []
    for j in range(1, num_sections + 1):
        for i in range(1, num_sections + 1):
            sections.append(
                (df["lon"][i - 1], df["lon"][i], df["lat"][j - 1], df["lat"][j])
            )
    return df, sections


num_sections = 3
data = get_raw_data(num_sections, 10)
visualize_test_data(data, num_sections)
