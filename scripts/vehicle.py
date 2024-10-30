'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-05-27 22:23:35
LastEditTime: 2024-10-31 01:01:38
FilePath: /vehicle-interaction-decision-making/scripts/vehicle.py
Copyright 2024 puyu, All Rights Reserved.
'''

import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from typing import Tuple, List, Union, Optional

import utils
from vehicle_base import VehicleBase
from planner import KLevelPlanner

current_dir_path = os.path.dirname(os.path.abspath(__file__))
vehicle_show_config = [
    # ["#30A9DE", f"{current_dir_path}/../img/vehicle/blue.png"],
    ["#0000FF", f"{current_dir_path}/../img/vehicle/blue.png"],
    ["#E53A40", f"{current_dir_path}/../img/vehicle/red.png"],
    ["#4CAF50", f"{current_dir_path}/../img/vehicle/green.png"],
    ["#FFFF00", f"{current_dir_path}/../img/vehicle/yellow.png"],
    ["#00FFFF", f"{current_dir_path}/../img/vehicle/cyan.png"],
    ["#FA58F4", f"{current_dir_path}/../img/vehicle/purple.png"],
    ["#000000", f"{current_dir_path}/../img/vehicle/black.png"],
]

class Vehicle(VehicleBase):
    global_vehicle_idx = 0

    def __init__(self, name, cfg: dict = {}) -> None:
        super().__init__(name)
        self.vehicle_box2d: np.ndarray = VehicleBase.get_box2d(self.state)
        self.safezone: np.ndarray = VehicleBase.get_safezone(self.state)
        self.target: utils.State = utils.State(0, 0, 0, 0)
        self.have_got_target: bool = False
        self.dt: float = cfg['delta_t']
        self.cur_action: Optional[utils.Action] = None
        self.excepted_traj: Optional[utils.StateList] = None
        self.footprint: List[utils.State] = []

        vehicle_info = cfg["vehicle_list"][name]
        self.level: int = vehicle_info["level"]
        self.init_x_min: float = vehicle_info["init"]["x"]["min"]
        self.init_x_max: float = vehicle_info["init"]["x"]["max"]
        self.init_y_min: float = vehicle_info["init"]["y"]["min"]
        self.init_y_max: float = vehicle_info["init"]["y"]["max"]
        self.init_v_min: float = vehicle_info["init"]["v"]["min"]
        self.init_v_max: float = vehicle_info["init"]["v"]["max"]
        self.init_yaw: float = vehicle_info["init"]["yaw"]
        self.target.x = vehicle_info["target"]["x"]
        self.target.y = vehicle_info["target"]["y"]
        self.target.yaw = vehicle_info["target"]["yaw"]
        self.vis_text_pos = utils.State(vehicle_info["text"]["x"], vehicle_info["text"]["y"])

        self.color: str = vehicle_show_config[Vehicle.global_vehicle_idx % len(vehicle_show_config)][0]
        vehicle_pic_path = vehicle_show_config[Vehicle.global_vehicle_idx % len(vehicle_show_config)][1]
        self.outlook = plt.imread(vehicle_pic_path, format = "png")
        Vehicle.global_vehicle_idx += 1

        self.planner = KLevelPlanner(cfg)

        self.reset()

    def reset(self):
        self.state.x = random.uniform(self.init_x_min, self.init_x_max)
        self.state.y = random.uniform(self.init_y_min, self.init_y_max)
        self.state.v = random.uniform(self.init_v_min, self.init_v_max)
        self.state.yaw = self.init_yaw

        self.cur_action = None
        self.excepted_traj = None
        self.have_got_target = False
        self.footprint = [self.state]

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

    def excute(self, others: List[VehicleBase]) -> Tuple[utils.Action, utils.StateList]:
        if self.is_get_target:
            self.have_got_target = True
            self.state.v = 0
            excepted_traj = utils.StateList()
            act = utils.Action.MAINTAIN
        else:
            act, excepted_traj = self.planner.planning(self, others)

        return act, excepted_traj

    def draw_vehicle(self, draw_style = 'realistic', fill_mode = False) -> None:
        if draw_style == 'realistic':
            transform_data = Affine2D().rotate_deg_around(
                self.state.x, self.state.y, self.state.yaw / np.pi * 180)
            transform_data += plt.gca().transData

            # 0.15 offset consider the length of the rearview mirror
            image_extent = [self.state.x - self.length / 2,
                            self.state.x + self.length / 2,
                            self.state.y - self.width / 2 - 0.15,
                            self.state.y + self.width / 2 + 0.15]
            plt.imshow(self.outlook, transform=transform_data,
                       extent=image_extent, zorder=10.0, clip_on=True)
        else:
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

    def reset(self):
        for vehicle in self.vehicle_list:
            vehicle.reset()

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
