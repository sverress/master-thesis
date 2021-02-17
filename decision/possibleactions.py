# Constants for cluster
number_of_scooters = 10
scooter_inventory = 4
battery_inventory = 5
ideal_state = 8

# Variables
pick_ups = max(number_of_scooters - ideal_state, 0)
swaps = min(number_of_scooters, battery_inventory)
drop_offs = max(min(ideal_state - number_of_scooters, scooter_inventory), 0)

combinations = []
for pick_up in range(pick_ups + 1):
    for swap in range(swaps + 1):
        for drop_off in range(drop_offs + 1):
            if (pick_up + swap) <= battery_inventory:
                combinations.append([pick_up, swap, drop_off])


combination = [3, 2, 0]
scooters = [i / 10 for i in range(1, 9)]
scooters_to_pick_up = scooters[: combination[0]]
scooters_to_swap = scooters[combination[0] : combination[0] + combination[1]]
print()
