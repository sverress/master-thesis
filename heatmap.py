import folium
import classes
import clustering.scripts
from folium import plugins
from folium.plugins import HeatMap


map_hooray = folium.Map(location=[59.925586, 10.730721], zoom_start=13)

world_to_analyse = classes.World(
    960,
    None,
    clustering.scripts.get_initial_state(
        2500,
        50,
        number_of_vans=2,
        number_of_bikes=0,
    ),
    verbose=False,
    visualize=False,
    MODELS_TO_BE_SAVED=5,
    TRAINING_SHIFTS_BEFORE_SAVE=50,
    ANN_LEARNING_RATE=0.0001,
    ANN_NETWORK_STRUCTURE=[1000, 2000, 1000, 200],
    REPLAY_BUFFER_SIZE=100,
    test_parameter_name="dr_ltr",
)
percentage = 1500 / 2500
all_coordinates = []
for cluster in world_to_analyse.state.clusters:
    cluster.scooters = cluster.scooters[: round(len(cluster.scooters) * percentage)]
    cluster.ideal_state = round(cluster.ideal_state * percentage)
    for scooter in cluster.scooters:
        all_coordinates.append(scooter.get_location())

HeatMap(all_coordinates).add_to(map_hooray)

# Display the map
map_hooray.save("heatmap.HTML")
