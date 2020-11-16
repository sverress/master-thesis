import pandas as pd
from model.BaseModel import BaseModel
from model.BaseModelInput import BaseModelInput
from math import log


class AlternativeModelInput(BaseModelInput):
    def __init__(
        self,
        scooter_list: pd.DataFrame,
        delivery_nodes_list: pd.DataFrame,
        depot_location: tuple,
        service_vehicles_dict: dict,
    ):
        self.Z = sorted(scooter_list.zone.unique())
        self.L_z = [
            list(
                scooter_list.loc[scooter_list["zone"] == i].index.union(
                    delivery_nodes_list.loc[delivery_nodes_list["zone"] == i].index
                )
            )
            for i in self.Z
        ]
        self.B = [0.0] + [
            x / 100 for x in scooter_list["battery"]
        ]  # Battery level of scooter at location i
        super().__init__(
            scooter_list, delivery_nodes_list, depot_location, service_vehicles_dict
        )

    def compute_reward_matrix(self, scooter_list, delivery_nodes_list):
        R_kz = {}
        alpha = 1.1
        beta = 1.3
        for z in self.Z:
            k_max = len(self.L_z[z])
            for k in range(0, k_max + 1):
                R_kz[(k, z)] = alpha * log(beta * k - 1)


class AlternativeModel(BaseModel):
    @staticmethod
    def get_input_class():
        return AlternativeModelInput

    def set_objective(self):
        pass
