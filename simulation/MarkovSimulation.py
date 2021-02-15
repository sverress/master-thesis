import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class MarkovSim:
    def __init__(
        self,
        transition_state=None,
        transition_porbs=None,
        zone_boundaries=None,
        nodes=None,
    ):

        self.instance = None  # quick access to nodes
        # [same position, zone 1, zone 2, etc..]
        self.transition_state = transition_state if transition_state else [0, 1, 2]
        # P(not moving, move to zone 1, move to zone 2, etc..)
        self.transition_probs = (
            transition_porbs if transition_porbs else [(0.5, 0.3, 0.2), (0.3, 0.5, 0.2)]
        )
        # (x_min, x_max, y_min, y_max)
        self.zone_boundaries = (
            zone_boundaries if zone_boundaries else [(0, 5, 0, 10), (5, 10, 0, 10)]
        )

        self.nodes = nodes if nodes else self.create_random_nodes(10)
        self.previous_nodes = []

    def transition(self):
        for node in self.nodes:
            self.previous_nodes.append(copy.deepcopy(node))

            trans_zone = np.random.choice(
                self.transition_state,
                replace=True,
                p=self.transition_probs[node.zone - 1],
            )

            if trans_zone != 0:
                x = np.random.uniform(
                    self.zone_boundaries[trans_zone - 1][0],
                    self.zone_boundaries[trans_zone - 1][1],
                )
                y = np.random.uniform(
                    self.zone_boundaries[trans_zone - 1][2],
                    self.zone_boundaries[trans_zone - 1][3],
                )
                node.previous_location = copy.deepcopy(node.location)
                node.location = (x, y)
                node.zone = trans_zone
            else:
                node.previous_location = copy.deepcopy(node.location)

    @staticmethod
    def visualize_transition(original_state, current_state, boundiries):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 9.7))
        ax1.axis("off")
        ax1.set_title("Before MC simulation", fontsize=16)
        ax2.axis("off")
        ax2.set_title("After MC simulation", fontsize=16)
        size = 400
        radius = np.sqrt(size) / 2.0
        alignment = "center"

        for i, node in enumerate(original_state):
            ax1.scatter(node.location[0], node.location[1], c="g", s=size, alpha=0.6)
            ax1.annotate(
                f"{i + 1}",
                (node.location[0], node.location[1]),
                horizontalalignment=alignment,
                verticalalignment=alignment,
            )

        for i, node in enumerate(current_state):
            if node.location == node.previous_location:
                ax2.scatter(
                    node.location[0], node.location[1], c="g", s=size, alpha=0.6,
                )

            else:
                x = node.location[0]
                y = node.location[1]
                x_p = node.previous_location[0]
                y_p = node.previous_location[1]

                ax2.scatter(
                    x, y, c="b", s=size, alpha=0.6,
                )
                ax2.scatter(
                    x_p, y_p, c="r", s=size, alpha=0.3,
                )

                arrow = mpl.patches.FancyArrowPatch(
                    posA=(x_p, y_p),
                    posB=(x, y),
                    arrowstyle="-|>",
                    mutation_scale=20,
                    shrinkA=radius,
                    shrinkB=radius,
                )
                ax2.add_patch(arrow)

            ax2.annotate(
                f"{i + 1}",
                (node.location[0], node.location[1]),
                horizontalalignment=alignment,
                verticalalignment=alignment,
            )

        for b in boundiries:
            x1, x2 = [b[0]] * 2, [b[1]] * 2
            y = b[2:]
            ax1.plot(x1, y, c="black")
            ax1.plot(x2, y, c="black")
            ax2.plot(x1, y, c="black")
            ax2.plot(x2, y, c="black")

        plt.show()

    def create_random_nodes(self, number_of_nodes):
        nodes = []
        for i in range(number_of_nodes):
            zone = np.random.randint(len(self.zone_boundaries))
            x = np.random.uniform(
                self.zone_boundaries[zone][0], self.zone_boundaries[zone][1]
            )
            y = np.random.uniform(
                self.zone_boundaries[zone][2], self.zone_boundaries[zone][3]
            )
            battery = np.random.uniform(0, 1)
            node = Node((x, y), battery, zone + 1)
            nodes.append(node)

        return nodes


class Node:
    def __init__(self, location, battery, zone):
        self.location = location
        self.battery = battery
        self.zone = zone
        self.previous_location = None


if __name__ == "__main__":
    mc = MarkovSim()
    mc.transition()
    mc.visualize_transition(mc.previous_nodes, mc.nodes, mc.zone_boundaries)
