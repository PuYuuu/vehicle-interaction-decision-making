import math
import random
import numpy as np
from collections import deque

import utils
from env import EnvCrossroads
from vehicle_base import VehicleBase


class Node:
     def __init__(self, x = 0, y = 0, yaw = 0, v = 0, action = None, p = None):
         self.x = x
         self.y = y
         self.yaw = yaw
         self.v = v
         self.action = action
         self.reward = 0
         self.parent = p


class KLevelPlanner:

    action_sets = [[0, 0],              # maintain
                   [0, math.pi / 6],    # turn left
                   [0, -math.pi / 6],   # turn right
                   [2.5, 0],            # accelerate
                   [-2.5, 0],           # decelerate
                   [-5, 0]]             # brake


    def __init__(self, env:EnvCrossroads, steps = 6, dt = 0.1, lamda = 0.9):
        self.env = env
        self.steps = steps
        self.dt = dt
        self.lamda = lamda
        self.weight_avoid = 1000
        self.weight_safe = 20
        self.weight_offroad = 200
        self.weight_velocity = 1000
        self.weight_direction = 10
        self.weight_distance = 10


    def planning(self, ego:VehicleBase, other:VehicleBase):
        other_prediction = self.get_prediction(ego, other)
        actions_id, traj, total_path = self.helper(ego, other, other_prediction)

        return KLevelPlanner.action_sets[actions_id[0]][0], KLevelPlanner.action_sets[actions_id[0]][1], traj, total_path


    def update(self, node:Node, act):
        acc = act[0]
        omega = act[1]

        x = node.x + node.v * np.cos(node.yaw) * self.dt
        y = node.y + node.v * np.sin(node.yaw) * self.dt
        v = node.v + acc * self.dt
        yaw = node.yaw + omega * self.dt

        if yaw > 2 * np.pi:
            yaw -= 2 * np.pi
        if yaw < 0:
            yaw += 2 * np.pi

        return x, y, yaw, v


    def helper(self, ego:VehicleBase, other:VehicleBase, traj):
        root = Node(ego.x, ego.y, ego.yaw, ego.v)
        queue = deque()
        queue.append(root)
        step = 0

        while len(queue) > 0:
            queue_len = len(queue)

            for _ in range(queue_len):
                tmp = queue.popleft()
                for i in range(len(KLevelPlanner.action_sets)):
                    if step > 0:
                        keep_rate = self.lamda ** step
                        random_number = random.random()
                        if random_number > keep_rate:
                            continue

                    x, y, yaw, v = self.update(tmp, KLevelPlanner.action_sets[i])
                    ego_box2d = ego.get_box2d([x, y])
                    ego_safezone = ego.get_safezone([x, y])

                    avoid = 0
                    if utils.has_overlap(ego_box2d, other.get_box2d(traj[step])):
                        avoid = -1
                    safe = 0
                    if utils.has_overlap(ego_safezone, other.get_safezone(traj[step])):
                        safe = -1
                    offroad = 0
                    for rect in self.env.rect:
                        if utils.has_overlap(ego_box2d, rect):
                            offroad = -1
                            break

                    direction = 0
                    if self.is_opposite_direction(x, y, yaw, ego_box2d):
                        direction = -1

                    velocity = 0
                    if velocity < 0:
                        velocity = -1

                    distance = -(abs(x - ego.target[0]) + abs(y - ego.target[1]) + 1.5 * abs(yaw - ego.target[2]))

                    cur_reward = self.weight_avoid * avoid +  self.weight_safe * safe + \
                                 self.weight_offroad * offroad + self.weight_distance * distance + \
                                 self.weight_direction * direction + self.weight_velocity * velocity

                    node = Node(x, y, yaw, v, i, tmp)
                    node.reward = tmp.reward + self.lamda ** step * cur_reward

                    queue.append(node)
            step += 1
            if step >= self.steps:
                break

        node = max(queue, key=lambda node:node.reward)
        action_id_list = []
        pos_list = []
        while node.parent != None:
            action_id_list.append(node.action)
            pos_list.append([node.x, node.y])
            node = node.parent

        # for debug
        total_candidate_path = []
        # for node in queue:
        #     path_tmp = [[], []]
        #     while node != None:
        #         path_tmp[0].append(node.x)
        #         path_tmp[1].append(node.y)
        #         node = node.parent
        #     total_candidate_path.append(path_tmp)
        # print("total candidate path: ", len(queue))
        actions = action_id_list[::-1]
        expected_traj = pos_list[::-1]

        return actions, expected_traj, total_candidate_path


    def get_prediction(self, ego: VehicleBase, other: VehicleBase):
        if ego.level == 0 or other.is_get_target:
            return [[other.x, other.y]] * self.steps
        elif ego.level == 1:
            other_prediction_ego = [[ego.x, ego.y]] * self.steps
            other_act, other_traj, _ = self.helper(other, ego, other_prediction_ego)
            return other_traj
        elif ego.level == 2:
            static_traj = [[other.x, other.y]] * self.steps
            _, ego_l0_traj = self.helper(ego, other, static_traj)
            _, other_l1_traj = self.helper(other, ego, ego_l0_traj)
            return other_l1_traj


    def is_opposite_direction(self, x, y, yaw, ego_box2d):
        for laneline in self.env.laneline:
            if utils.has_overlap(ego_box2d, laneline):
                return True

        lanewidth = self.env.lanewidth

        # down lane
        if x > -lanewidth and x < 0 and (y < -lanewidth or y > lanewidth):
            if yaw > 0 and yaw < np.pi:
                return True
        # up lane
        elif x > 0 and x < lanewidth and (y < -lanewidth or y > lanewidth):
            if not (yaw > 0 and yaw < np.pi):
                return True
        # right lane
        elif y > -lanewidth and y < 0 and (x < -lanewidth or x > lanewidth):
            if yaw > 0.5 * np.pi and yaw < 1.5 * np.pi:
                return True
        # left lane
        elif y > 0 and y < lanewidth and (x < -lanewidth or x > lanewidth):
            if not (yaw > 0.5 * np.pi and yaw < 1.5 * np.pi):
                return True

        return False
