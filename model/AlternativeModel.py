import pandas as pd
from gurobipy import GRB
import gurobipy as gp

from instance.InstanceManager import InstanceManager
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
        T_max: int,
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
            scooter_list,
            delivery_nodes_list,
            depot_location,
            service_vehicles_dict,
            T_max,
        )

    def compute_reward_matrix(self, scooter_list, delivery_nodes_list):
        r_kz = {}
        alpha = 1.1
        beta = 1.3
        for z in self.Z:
            k_max = len(self.L_z[z])
            for k in range(0, k_max + 1):
                r_kz[(k, z)] = alpha * log(beta * k + 1) + 1
        return r_kz


class AlternativeModel(BaseModel):
    def __init__(self, model_input, setup=True, time_limit=None):
        # x_ijv - 1 if, for service vehicle v, visit to location i is followed by a visit to location j- 0 otherwise
        super().__init__(model_input, setup=False, time_limit=time_limit)
        self.cart_k_z = [
            (k, z) for z in self._.Z for k in range(len(self._.L_z[z]) + 1)
        ]
        self.w = self.m.addVars(self.cart_k_z, vtype=GRB.BINARY, name="w")
        self.setup()

    @staticmethod
    def get_input_class():
        return AlternativeModelInput

    def set_objective(self):
        self.m.setObjective(
            gp.quicksum(
                self._.reward[(k, z)] * self.w[(k, z)] for k, z in self.cart_k_z
            )
            - gp.quicksum(
                self._.B[i] * self.y[(i, v)] for i, v in self.cart_loc_v_scooters
            ),
            GRB.MAXIMIZE,
        )

    def setup(self):
        # Adding the constraints and call the set_objective function
        super().setup()
        # Adding the additional constraints
        self.m.addConstrs(
            (
                gp.quicksum(k * self.w[(k, z)] for k in range(len(self._.L_z[z]) + 1))
                == gp.quicksum(
                    self.y[(i, v)]
                    for v in self._.service_vehicles
                    for i in self._.L_z[z]
                )
                - gp.quicksum(
                    self.p[(i, v)]
                    for v in self._.service_vehicles
                    for i in self._.L_z[z]
                    if i in self._.scooters
                )
                for z in self._.Z
            ),
            "force_w_1",
        )
        self.m.addConstrs(
            (
                gp.quicksum(self.w[(k, z)] for k in range(len(self._.L_z[z]) + 1)) <= 1
                for z in self._.Z
            ),
            "force_w_2",
        )


if __name__ == "__main__":
    manager = InstanceManager()
    instance = manager.create_test_instance(2, 2, AlternativeModel)
    instance.run()
    instance.visualize_raw_data_map()
    instance.model.print_solution()
