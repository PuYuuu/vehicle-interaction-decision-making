import numpy as np
import matplotlib.pyplot as plt

import utils
from vehicle_base import VehicleBase
from planner import KLevelPlanner


class Vehicle(VehicleBase):
    def __init__(self, name, x, y, yaw, env = None, dt = 0.1, color = 'k', length = 5,
                 width = 2, safe_length = 8, safe_width = 2.4):
        self.name = name
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = 0
        self.length = length
        self.width = width
        self.color = color
        self.safe_length = safe_length
        self.safe_width = safe_width
        self.vehicle_box2d = self.get_box2d()
        self.safezone = self.get_safezone()
        self.level = 0
        self.target = [0, 0, 0]
        self.have_got_target = False
        self.dt = dt

        self.env = env
        self.planner = KLevelPlanner(env, 6, self.dt)


    def set_level(self, level):
        if level >= 0 and level < 3:
            self.level = level
        else:
            print("set_level error, the level must be >= 0 and > 3 !")


    def set_target(self, target):
        if len(target) != 3:
            print("set_target error, the target len must equal 3 !")
            return

        if target[0] >= -25 and target[0] <= 25 and target[1] >= -25 and target[1] <= 25:
            self.target = target
        else:
            print("set_target error, the target range must >= -25 and <= 25 !")


    def get_box2d(self, tar_offset = None):
        vehicle = np.array(
            [[-self.length/2, self.length/2, self.length/2, -self.length/2, -self.length/2],
            [self.width/2, self.width/2, -self.width/2, -self.width/2, self.width/2]]
        )
        rot = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                     [np.sin(self.yaw), np.cos(self.yaw)]])

        vehicle = np.dot(rot, vehicle)
        if tar_offset is None:
            vehicle += np.array([[self.x], [self.y]])
        else:
            vehicle += np.array([[tar_offset[0]], [tar_offset[1]]])

        return vehicle


    def get_safezone(self, tar_offset = None):
        safezone = np.array(
            [[-self.safe_length/2, self.safe_length/2, self.safe_length/2, -self.safe_length/2, -self.safe_length/2],
            [self.safe_width/2, self.safe_width/2, -self.safe_width/2, -self.safe_width/2, self.safe_width/2]]
        )
        rot = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                     [np.sin(self.yaw), np.cos(self.yaw)]])

        safezone = np.dot(rot, safezone)
        if tar_offset is None:
            safezone += np.array([[self.x], [self.y]])
        else:
            safezone += np.array([[tar_offset[0]], [tar_offset[1]]])

        return safezone


    def excute(self, other:VehicleBase):
        acc, omega, excepted_traj = self.planner.planning(self, other)

        # update ego
        tmp_node = utils.Node(self.x, self.y, self.yaw, self.v)
        self.x, self.y, self.yaw, self.v = utils.kinematic_propagate(
            tmp_node, [acc, omega], self.length * 0.8, self.dt)

        if ((self.x - self.target[0]) ** 2 + (self.y - self.target[1]) ** 2) < 3:
            self.have_got_target = True
            self.v = 0
            excepted_traj = []

        return excepted_traj


    def draw_vehicle(self, fill_mode = False):
        head = np.array(
            [[0.3 * self.length, 0.3 * self.length], [self.width/2, -self.width/2]]
        )
        rot = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                     [np.sin(self.yaw), np.cos(self.yaw)]])
        head = np.dot(rot, head)
        head += np.array([[self.x], [self.y]])

        self.vehicle_box2d = self.get_box2d()

        if not fill_mode:
            plt.plot(self.vehicle_box2d[0, :], self.vehicle_box2d[1, :], self.color)
            plt.plot(head[0, :], head[1, :], self.color)
        else:
            plt.fill(self.vehicle_box2d[0, :], self.vehicle_box2d[1, :], color=self.color, alpha=0.5)


    @property
    def is_get_target(self):
        return self.have_got_target

