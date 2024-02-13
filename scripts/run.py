import math
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


    def __init__(self, env:EnvCrossroads, level = 0, steps = 5, dt = DT, lamda = 0.9):
        self.env = env
        self.level = level
        self.steps = steps
        self.dt = dt
        self.lamda = lamda
        self.weight_avoid = 200
        self.weight_safe = 20
        self.weight_offroad = 100
        self.weight_direction = 10
        self.weight_distance = 1


    def planning(self, ego:Vehicle, other:Vehicle):
        root = Node(ego.x, ego.y, ego.yaw, ego.v)
        queue = deque()
        queue.append(root)

        other_prediction = self.get_prediction(ego, other)

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
                    if Vehicle.has_overlap(ego_box2d, other.get_box2d(other_prediction[step])):
                        avoid = -1
                    safe = 0
                    if Vehicle.has_overlap(ego_safezone, other.get_safezone(other_prediction[step])):
                        safe = -1
                    offroad = 0
                    for rect in self.env.rect:
                        if Vehicle.has_overlap(ego_box2d, rect):
                            offroad = -1
                            break
                    direction = 0
                    distance = -(abs(x - ego.target[0]) + abs(y - ego.target[1]))

                    cur_reward = self.weight_avoid * avoid +  self.weight_safe * safe + \
                                 self.weight_offroad * offroad + self.weight_distance * distance + \
                                 self.weight_direction * direction

                    node = Node(x, y, yaw, v, i, tmp)
                    node.reward = tmp.reward + self.lamda ** step * cur_reward
                    queue.append(node)
            step += 1
            if step >= self.steps:
                break
        node = max(queue, key=lambda node:node.reward)
        action_id = node.action
        while node.parent != None:
            action_id = node.action
            node = node.parent

        return KLevelPlanner.action_sets[action_id][0], KLevelPlanner.action_sets[action_id][1]


    def update(self, node:Node, act):
        acc = act[0]
        omega = act[1]

        x = node.x + node.v * np.cos(node.yaw) * self.dt
        y = node.y + node.v * np.sin(node.yaw) * self.dt
        v = node.v + acc * self.dt
        yaw = node.yaw + omega * self.dt

        return x, y, yaw, v


    def get_prediction(self, ego: Vehicle, other: Vehicle):
        
        if ego.level == 0:
            return [[other.x, other.y]] * self.steps
        elif ego.level == 1:
            pass
        elif ego.level == 2:
            pass


def run():
    env = EnvCrossroads(size=25, lanewidth=4)
    vehicle_0 = Vehicle(2, -16, np.pi / 2, 'blue')  # the straight vehicle
    vehicle_1 = Vehicle(-2, 16, -np.pi / 2, 'red')  # the turn left vehicle 

    vehicle_0.set_level(0)
    vehicle_1.set_level(0)
    vehicle_0.set_target([2, 16])
    vehicle_1.set_target([15, -2])

    planner = KLevelPlanner(env)

    while True:
        if np.hypot(vehicle_1.target[0] - vehicle_1.x, vehicle_1.target[1] - vehicle_1.y) < 1 and \
           np.hypot(vehicle_0.target[0] - vehicle_0.x, vehicle_0.target[1] - vehicle_0.y) < 1:
            print("success !")
            break

        if Vehicle.has_overlap(vehicle_1.get_box2d(), vehicle_0.get_box2d()):
            print("failed, exit...")
            break

        acc, omega = planner.planning(vehicle_1, vehicle_0)
        vehicle_1.update(acc, omega, DT)

        acc, omega = planner.planning(vehicle_0, vehicle_1)
        vehicle_0.update(acc, omega, DT)

        plt.cla()
        env.draw_env()
        vehicle_0.draw_vehicle()
        vehicle_1.draw_vehicle()
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.gca().set_aspect('equal')
        plt.pause(0.1)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    run()

