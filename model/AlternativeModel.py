from math import log
from gurobipy import GRB
import gurobipy as gp
from model.BaseModel import BaseModel
from model.BaseModelInput import BaseModelInput


class AlternativeModelInput(BaseModelInput):
    def compute_reward_matrix(self, scooter_list, delivery_nodes_list):
        r_kz = {}
        alpha = 1.1
        beta = 1.3
        for z in self.zones:
            k_max = len(self.zone_scooters[z])
            for k in range(0, k_max + 1):
                r_kz[(k, z)] = alpha * log(beta * k + 1) + 1
        return r_kz


class AlternativeModel(BaseModel):
    def __init__(self, model_input, setup=True, time_limit=None, **kwargs):
        # x_ijv - 1 if, for service vehicle v, visit to location i is followed by a visit to location j- 0 otherwise
        super().__init__(model_input, setup=False, time_limit=time_limit)
        self.cart_k_z = [
            (k, z)
            for z in self._.zones
            for k in range(len(self._.zone_scooters[z]) + 1)
        ]
        self.w = self.m.addVars(self.cart_k_z, vtype=GRB.BINARY, name="w")
        self.setup()

    def to_string(self, short_name=True):
        return "A" if short_name else "Alternative"

    @staticmethod
    def get_input_class():
        return AlternativeModelInput

    def set_objective(self):
        self.m.setObjective(
            gp.quicksum(
                self._.reward[(k, z)] * self.w[(k, z)] for k, z in self.cart_k_z
            )
            - gp.quicksum(
                self._.battery_level[i] * self.y[(i, v)]
                for i, v in self.cart_loc_v_scooters
            ),
            GRB.MAXIMIZE,
        )

    def setup(self):
        # Adding the constraints and call the set_objective function
        super().setup()
        # Adding the additional constraints
        self.m.addConstrs(
            (
                gp.quicksum(
                    k * self.w[(k, z)] for k in range(len(self._.zone_scooters[z]) + 1)
                )
                == gp.quicksum(
                    self.y[(i, v)]
                    for v in self._.service_vehicles
                    for i in self._.zone_scooters[z]
                )
                - gp.quicksum(
                    self.p[(i, v)]
                    for v in self._.service_vehicles
                    for i in self._.zone_scooters[z]
                    if i in self._.scooters
                )
                for z in self._.zones
            ),
            "force_w_1",
        )
        self.m.addConstrs(
            (
                gp.quicksum(
                    self.w[(k, z)] for k in range(len(self._.zone_scooters[z]) + 1)
                )
                <= 1
                for z in self._.zones
            ),
            "force_w_2",
        )
