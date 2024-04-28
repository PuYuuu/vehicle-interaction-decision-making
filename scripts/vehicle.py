import logging
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, List

import utils
from vehicle_base import VehicleBase
from planner import KLevelPlanner


class Vehicle(VehicleBase):
    def __init__(self, name, state, color = 'k', cfg: dict = {}):
        super().__init__(name, state, color)
        self.vehicle_box2d = VehicleBase.get_box2d(self.state)
        self.safezone = VehicleBase.get_safezone(self.state)
        self.level = 0
        self.target = utils.State(0, 0, 0, 0)
        self.have_got_target = False
        self.dt = cfg['delta_t']

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

    def excute(self, other: VehicleBase) -> Tuple[utils.Action, List]:
        if self.is_get_target:
            self.have_got_target = True
            self.state.v = 0
            excepted_traj = []
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
