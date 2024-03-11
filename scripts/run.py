import math
import time
import random
import threading
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from env import EnvCrossroads
from vehicle import Vehicle

DT = 0.5

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
                   [0, math.pi / 4],    # turn left
                   [0, -math.pi / 4],   # turn right
                   [2.5, 0],            # accelerate
                   [-2.5, 0],           # decelerate
                   [-5, 0]]             # brake


    def __init__(self, env:EnvCrossroads, level = 0, steps = 6, dt = DT, lamda = 0.9):
        self.env = env
        self.level = level
        self.steps = steps
        self.dt = dt
        self.lamda = lamda
        self.weight_avoid = 200
        self.weight_safe = 20
        self.weight_offroad = 100
        self.weight_direction = 10
        self.weight_velocity = 10
        self.weight_distance = 1


    def planning(self, ego:Vehicle, other:Vehicle):
        other_prediction = self.get_prediction(ego, other)
        actions_id, _ =  self.helper(ego, other, other_prediction)

        return KLevelPlanner.action_sets[actions_id[0]][0], KLevelPlanner.action_sets[actions_id[0]][1]


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


    def helper(self, ego:Vehicle, other:Vehicle, traj):
        root = Node(ego.x, ego.y, ego.yaw, ego.v)
        queue = deque()
        queue.append(root)
        step = 0

        while len(queue) > 0:
            queue_len = len(queue)

            for _ in range(queue_len):
                tmp = queue.popleft()
                for i in range(len(KLevelPlanner.action_sets)):
                    x, y, yaw, v = self.update(tmp, KLevelPlanner.action_sets[i])
                    ego_box2d = ego.get_box2d([x, y])
                    ego_safezone = ego.get_safezone([x, y])

                    avoid = 0
                    if Vehicle.has_overlap(ego_box2d, other.get_box2d(traj[step])):
                        avoid = -1
                    safe = 0
                    if Vehicle.has_overlap(ego_safezone, other.get_safezone(traj[step])):
                        safe = -1
                    offroad = 0
                    for rect in self.env.rect:
                        if Vehicle.has_overlap(ego_box2d, rect):
                            offroad = -1
                            break
                    direction = 0
                    if self.is_opposite_direction(x, y, yaw, ego_box2d):
                        direction = -1

                    velocity = 0
                    if v < -0.1:
                        velocity = -1

                    distance = -(abs(x - ego.target[0]) + abs(y - ego.target[1]))

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

        actions = action_id_list[::-1]
        expected_traj = pos_list[::-1]

        return actions, expected_traj


    def get_prediction(self, ego: Vehicle, other: Vehicle):
        
        if ego.level == 0:
            return [[other.x, other.y]] * self.steps
        elif ego.level == 1:
            other_prediction_ego = [[ego.x, ego.y]] * self.steps
            other_act, other_traj = self.helper(other, ego, other_prediction_ego)
            return other_traj
        elif ego.level == 2:
            static_traj = [[other.x, other.y]] * self.steps
            _, ego_l0_traj = self.helper(ego, other, static_traj)
            _, other_l1_traj = self.helper(other, ego, ego_l0_traj)
            return other_l1_traj


    def is_opposite_direction(self, x, y, yaw, ego_box2d):
        for laneline in self.env.laneline:
            if Vehicle.has_overlap(ego_box2d, laneline):
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


def run():
    env = EnvCrossroads(size=25, lanewidth=4.2)
    max_per_iters = 20 / DT

    succeed_count = 0
    for iter in range(1):
        init_y_0 = random.uniform(-20, -12)
        init_y_1 = random.uniform(12, 20)
        init_v_0 = random.uniform(3, 5)
        init_v_1 = random.uniform(3, 5)

        # the turn left vehicle
        vehicle_0 = Vehicle("vehicle_0", env.lanewidth / 2, init_y_0, np.pi / 2, 'blue')
        # the straight vehicle
        vehicle_1 = Vehicle("vehicle_1", -env.lanewidth / 2, init_y_1, -np.pi / 2, 'red')

        vehicle_0.set_level(0)
        vehicle_1.set_level(1)
        vehicle_0.set_target([-15, env.lanewidth / 2])
        vehicle_1.set_target([-env.lanewidth / 2, -15])
        vehicle_0.v = init_v_0
        vehicle_1.v = init_v_1

        planner = KLevelPlanner(env)

        vehicle_0_history = [[env.lanewidth / 2, init_y_0, np.pi / 2]]
        vehicle_1_history = [[-env.lanewidth / 2, init_y_1, -np.pi / 2]]

        cur_loop_count = 0
        while True:
            if vehicle_0.is_get_target and vehicle_1.is_get_target:
                print("success !")
                succeed_count += 1
                break

            if Vehicle.has_overlap(vehicle_1.get_box2d(), vehicle_0.get_box2d()) or \
                cur_loop_count > max_per_iters:
                print("failed, exit...")
                break

            # TODO: Use multithreading to optimize time cost
            start_time = time.time()
            if not vehicle_0.is_get_target:
                acc, omega = planner.planning(vehicle_0, vehicle_1)
                vehicle_0.update(acc, omega, DT)
                vehicle_0_history.append([vehicle_0.x, vehicle_0.y, vehicle_0.yaw])
 
            if not vehicle_1.is_get_target:
                acc, omega = planner.planning(vehicle_1, vehicle_0)
                vehicle_1.update(acc, omega, DT)
                vehicle_1_history.append([vehicle_1.x, vehicle_1.y, vehicle_1.yaw])
            elapsed_time = time.time() - start_time
            print("cost time: {}".format(elapsed_time))

            plt.cla()
            env.draw_env()
            vehicle_0.draw_vehicle()
            vehicle_1.draw_vehicle()
            plt.plot(vehicle_0.target[0], vehicle_0.target[1], "xb")
            plt.plot(vehicle_1.target[0], vehicle_1.target[1], "xr")
            plt.xlim(-25, 25)
            plt.ylim(-25, 25)
            plt.gca().set_aspect('equal')
            plt.pause(0.1)
            cur_loop_count += 1
        
        plt.cla()
        env.draw_env()
        for history in vehicle_0_history:
            tmp = Vehicle("tmp", history[0], history[1], history[2], "blue")
            tmp.draw_vehicle(False)
        for history in vehicle_1_history:
            tmp = Vehicle("tmp", history[0], history[1], history[2], "red")
            tmp.draw_vehicle(False)
        plt.plot(vehicle_0.target[0], vehicle_0.target[1], "xb")
        plt.plot(vehicle_1.target[0], vehicle_1.target[1], "xr")
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.gca().set_aspect('equal')
        plt.pause(60)
    print(succeed_count)


if __name__ == "__main__":
    run()
