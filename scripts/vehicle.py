import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional

import utils
from vehicle_base import VehicleBase
from planner import KLevelPlanner


class Vehicle(VehicleBase):
    def __init__(self, name, state, color = 'k', cfg: dict = {}) -> None:
        super().__init__(name, state, color)
        self.vehicle_box2d: np.ndarray = VehicleBase.get_box2d(self.state)
        self.safezone: np.ndarray = VehicleBase.get_safezone(self.state)
        self.level: int = 0
        self.target: utils.State = utils.State(0, 0, 0, 0)
        self.have_got_target: bool = False
        self.dt: float = cfg['delta_t']
        self.cur_action: Optional[utils.Action] = None
        self.excepted_traj: Optional[utils.StateList] = None
        self.footprint: List[utils.State] = [state]

        self.planner = KLevelPlanner(cfg)

    def set_level(self, level) -> None:
        if level >= 0 and level < 3:
            self.level = level
        else:
            logging.CRITICAL("set_level error, the level must be >= 0 and > 3 !")

    def set_target(self, target: utils.State) -> None:
        if target.x >= -25 and target.x <= 25 and target.y >= -25 and target.y <= 25:
            self.target = target
        else:
            logging.CRITICAL("set_target error, the target range must >= -25 and <= 25 !")

    def excute(self, other: VehicleBase) -> Tuple[utils.Action, utils.StateList]:
        if self.is_get_target:
            self.have_got_target = True
            self.state.v = 0
            excepted_traj = utils.StateList()
            act = utils.Action.MAINTAIN
        else:
            act, excepted_traj = self.planner.planning(self, other)

        return act, excepted_traj

    def draw_vehicle(self, fill_mode = False) -> None:
        head = np.array(
            [[0.3 * VehicleBase.length, 0.3 * VehicleBase.length],
             [VehicleBase.width/2, -VehicleBase.width/2]])
        rot = np.array([[np.cos(self.state.yaw), -np.sin(self.state.yaw)],
                        [np.sin(self.state.yaw), np.cos(self.state.yaw)]])
        head = np.dot(rot, head)
        head += np.array([[self.state.x], [self.state.y]])

        self.vehicle_box2d = VehicleBase.get_box2d(self.state)

        if not fill_mode:
            plt.plot(self.vehicle_box2d[0, :], self.vehicle_box2d[1, :], self.color)
            plt.plot(head[0, :], head[1, :], self.color)
        else:
            plt.fill(self.vehicle_box2d[0, :], self.vehicle_box2d[1, :],
                     color=self.color, alpha=0.5)

    @property
    def is_get_target(self) -> bool:
        return self.have_got_target or \
               ((self.state.x - self.target.x) ** 2 + (self.state.y - self.target.y) ** 2) < 3


class VehicleList:
    def __init__(self, vehicle_list = None) -> None:
        self.vehicle_list: List[Vehicle] = vehicle_list if vehicle_list is not None else []

    @property
    def is_all_get_target(self) -> bool:
        return all(vehicle.is_get_target for vehicle in self.vehicle_list)

    @property
    def is_any_collision(self) -> bool:
        for i in range(len(self.vehicle_list) - 1):
            for j in range(i + 1, len(self.vehicle_list)):
                if utils.has_overlap(
                    VehicleBase.get_box2d(self.vehicle_list[i].state),
                    VehicleBase.get_box2d(self.vehicle_list[j].state)):
                    return True

        return False

    def append(self, vehicle: Vehicle) -> None:
        self.vehicle_list.append(vehicle)

    def exclude(self, ego: Union[int, Vehicle]) -> List[Vehicle]:
        if isinstance(ego, int):
            return [item for idx, item in enumerate(self.vehicle_list) if idx != ego]
        elif isinstance(ego, Vehicle):
            return [vec for vec in self.vehicle_list if vec is not ego]
        else:
            logging.warning(f"VehicleList.exclude input type must be int or Vehicle")
            return []

    def __len__(self) -> int:
        return len(self.vehicle_list)

    def __getitem__(self, key: int) -> Vehicle:
        return self.vehicle_list[key]

    def __setitem__(self, key: int, value: Vehicle) -> None:
        self.vehicle_list[key] = value
