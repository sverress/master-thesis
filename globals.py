"""
WORLD SETTINGS
"""
# GEOSPATIAL DATA
GEOSPATIAL_BOUND_NEW = (59.9040, 59.9547, 10.6478, 10.8095)
CLUSTER_CENTER_DELTA = 0.001

# Inventory of vehicle
VAN_BATTERY_INVENTORY = 50
VAN_SCOOTER_INVENTORY = 20
BIKE_BATTERY_INVENTORY = 20
BIKE_SCOOTER_INVENTORY = 0

# Speed of service vehicles
VEHICLE_SPEED = 20
MINUTES_IN_HOUR = 60

# Speed of scooter ref - Fearnley at al. (2020, section 3.6)
SCOOTER_SPEED = 7

# Depot parameters
MAIN_DEPOT_LOCATION = (59.931794, 10.788314)
SMALL_DEPOT_LOCATIONS = [(59.908009, 10.741604), (59.944473, 10.748624)]
MAIN_DEPOT_CAPACITY = 10000
SMALL_DEPOT_CAPACITY = 100
CHARGE_TIME_PER_BATTERY = 60
SWAP_TIME_PER_BATTERY = 0.4

"""
DECISION PARAMETERS
"""
DISCOUNT_RATE = 0.80
EPSILON = 0.1  # Probability of taking a random action

# LINEAR MODEL
WEIGHT_UPDATE_STEP_SIZE = 0.001
VEHICLE_INVENTORY_STEP_SIZE = 0.25
WEIGHT_INITIALIZATION_VALUE = 0.0

# Default simulation constants
ITERATION_LENGTH_MINUTES = 20
NUMBER_OF_ROLLOUTS = 10
BATTERY_LIMIT = 20.0

# Negative reward for lost trips
LOST_TRIP_REWARD = -0.1

# Testing parameters
DEFAULT_NUMBER_OF_NEIGHBOURS = 3

"""
IMPORTANT PATHS
"""

STATE_CACHE_DIR = "test_state_cache"

# Test data directory
TEST_DATA_DIRECTORY = "test_data"

"""
VISUALIZATION
"""
# Colors for visualizer
BLUE, GREEN, RED, BLACK, WHITE = "blue", "green", "red", "black", "white"
VEHICLE_COLORS = ["blue", "green", "black", "purple"]

ACTION_OFFSET = 0.018

COLORS = [
    "#B52CC2",
    "#EF9A90",
    "#43C04E",
    "#CDCAE9",
    "#C00283",
    "#02BE50",
    "#D69F94",
    "#A4847E",
    "#CE058B",
    "#39029A",
    "#9EEB33",
    "#056672",
    "#FC726E",
    "#8C109C",
    "#D8FB27",
    "#BBE5D1",
    "#FEEB81",
    "#126027",
    "#7666E7",
    "#530788",
    "#A281ED",
    "#954701",
    "#B42760",
    "#F0E466",
    "#A32315",
    "#4886E8",
    "#117427",
    "#A3A66A",
    "#F124AC",
    "#4572BD",
    "#93EB5F",
    "#ECDCCD",
    "#48317F",
    "#DF8547",
    "#1DE961",
    "#5BD669",
    "#4FAA9B",
    "#937016",
    "#840FF6",
    "#3EAEFD",
    "#F6F34D",
    "#015133",
    "#59025B",
    "#F03B29",
    "#53A912",
    "#34058C",
    "#FA928D",
    "#3C70C3",
    "#AB9869",
    "#B6BD37",
    "#693C24",
    "#2588F7",
    "#54B006",
    "#6604CE",
    "#4A4329",
    "#0175B1",
    "#177982",
    "#544FAD",
    "#DD5409",
    "#583ED1",
    "#CD9D69",
    "#6B0BCE",
    "#D14B12",
    "#96725D",
    "#BB137F",
    "#7B53B5",
    "#BFFB24",
    "#F9D08F",
    "#CF03B8",
    "#A6F591",
    "#D7CFDB",
    "#2D4AD6",
    "#BC5286",
    "#6245C8",
    "#E40EB7",
    "#E2DA97",
    "#EE5089",
    "#CAF026",
    "#668981",
    "#8E424B",
    "#49633D",
    "#8A4CE4",
    "#827C33",
    "#35EFF2",
    "#325041",
    "#2BC23F",
    "#44857A",
    "#DA0043",
    "#87A43F",
    "#D4FCEC",
    "#9FD87C",
    "#0D36DF",
    "#241B73",
    "#524526",
    "#163F53",
    "#4C9B58",
    "#00F4DB",
    "#20054B",
    "#82026F",
    "#CA561D",
    "#F94B06",
    "#5CCBDB",
    "#8B6882",
    "#9C28B0",
    "#15357B",
    "#BB00F4",
    "#451918",
    "#B94AE1",
    "#698290",
    "#415697",
    "#61B95D",
    "#957BD8",
    "#01A1C5",
    "#69E54F",
    "#D40C21",
    "#08A810",
    "#05ECC3",
    "#8FA2B5",
    "#D45A2C",
    "#1689EA",
    "#7DD21F",
    "#A615B6",
    "#430E4C",
    "#557F16",
    "#68E3A4",
    "#E19180",
    "#8B0197",
    "#7314C4",
    "#A397DA",
    "#175ACE",
    "#6185AD",
    "#D981A8",
    "#984ED3",
    "#37FFF0",
    "#90BB50",
    "#A818B0",
    "#28F263",
    "#700EA8",
    "#5C0D3A",
    "#CAF06F",
    "#815F36",
    "#CCF509",
    "#21C91D",
    "#D09B45",
    "#282AF6",
    "#053525",
    "#0FAE75",
    "#213E02",
    "#1572AA",
    "#9D9A3A",
    "#1C1DA9",
    "#C6A728",
    "#0BE59B",
    "#272CAF",
    "#75BA93",
    "#E29981",
    "#45F101",
    "#D8BA19",
    "#BF7545",
    "#0F85B1",
    "#E6DC7B",
    "#6B6548",
    "#78B075",
    "#AFDF4D",
    "#D0BD94",
    "#C6F81B",
    "#27C209",
    "#3C6574",
    "#2CE0B3",
    "#9C6E06",
    "#53CECD",
    "#A5EC06",
    "#AA83D6",
    "#7705D2",
    "#806015",
    "#881E9E",
    "#617730",
    "#1F9ACF",
    "#8AE30F",
    "#D1E1B4",
    "#D924F6",
    "#5FE267",
    "#6BDDF2",
    "#5E40A5",
    "#9B1580",
    "#B6E49C",
    "#619C46",
    "#504BDE",
]
