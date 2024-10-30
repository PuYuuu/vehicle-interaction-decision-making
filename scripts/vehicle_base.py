'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-04-27 16:17:27
LastEditTime: 2024-10-31 01:01:31
FilePath: /vehicle-interaction-decision-making/scripts/vehicle_base.py
Copyright 2024 puyu, All Rights Reserved.
'''

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from utils import State
from env import EnvCrossroads


class VehicleBase(ABC):
    length = 5
    width = 2
    safe_length = 8
    safe_width = 2.4
    env: Optional[EnvCrossroads] = None

    def __init__(self, name: str):
        self.name: str = name
        self.state: State = State()

    @staticmethod
    def get_box2d(tar_offset: State) -> np.ndarray:
        vehicle = np.array(
            [[-VehicleBase.length/2, VehicleBase.length/2,
              VehicleBase.length/2, -VehicleBase.length/2, -VehicleBase.length/2],
            [VehicleBase.width/2, VehicleBase.width/2,
             -VehicleBase.width/2, -VehicleBase.width/2, VehicleBase.width/2]]
        )
        rot = np.array([[np.cos(tar_offset.yaw), -np.sin(tar_offset.yaw)],
                     [np.sin(tar_offset.yaw), np.cos(tar_offset.yaw)]])

        vehicle = np.dot(rot, vehicle)
        vehicle += np.array([[tar_offset.x], [tar_offset.y]])

        return vehicle

    @staticmethod
    def get_safezone(tar_offset: State) -> np.ndarray:
        safezone = np.array(
            [[-VehicleBase.safe_length/2, VehicleBase.safe_length/2,
              VehicleBase.safe_length/2, -VehicleBase.safe_length/2, -VehicleBase.safe_length/2],
            [VehicleBase.safe_width/2, VehicleBase.safe_width/2,
             -VehicleBase.safe_width/2, -VehicleBase.safe_width/2, VehicleBase.safe_width/2]]
        )
        rot = np.array([[np.cos(tar_offset.yaw), -np.sin(tar_offset.yaw)],
                     [np.sin(tar_offset.yaw), np.cos(tar_offset.yaw)]])

        safezone = np.dot(rot, safezone)
        safezone += np.array([[tar_offset.x], [tar_offset.y]])

        return safezone

    @staticmethod
    def initialize(env: EnvCrossroads, len: float, width: float,
                   safe_len: float, safe_width: float):
        VehicleBase.env = env
        VehicleBase.length = len
        VehicleBase.width = width
        VehicleBase.safe_length = safe_len
        VehicleBase.safe_width = safe_width
