import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from env import EnvCrossroads
from vehicle import Vehicle

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


    def __init__(self, level = 0, steps = 6, dt = 0.5, lamda = 0.9):
        self.level = level
        self.steps = steps
        self.dt = dt
        self.lamda = lamda


    def planning(self, vehicle:Vehicle, target:list):
        root = Node(vehicle.x, vehicle.y, vehicle.yaw, vehicle.v)
        queue = deque()
        queue.append(root)

        step = 0
        while len(queue) > 0:
            queue_len = len(queue)
            step += 1
            for _ in range(queue_len):
                tmp = queue.popleft()
                for i in range(len(KLevelPlanner.action_sets)):
                    x, y, yaw, v = self.update(tmp, KLevelPlanner.action_sets[i])
                    if ((-25 <= x <= -4 and 4 <= y <= 25) or
                        (4 <= x <= 25 and 4 <= y <= 25) or
                        (-25 <= x <= -4 and -25 <= y <= -4) or
                        (4 <= x <= 25 and -25 <= y <= -4)):
                        continue
                    node = Node(x, y, yaw, v, i, tmp)
                    node.reward = tmp.reward - (abs(x - target[0]) + abs(y - target[1]))
                    queue.append(node)
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


def run():
    env = EnvCrossroads(size=25, lanewidth=4)
    vehicle_0 = Vehicle(2, -16, np.pi / 2, 5, 2, 'blue')
    vehicle_1 = Vehicle(-2, 16, -np.pi / 2, 5, 2, 'red')

    planner = KLevelPlanner()

    while True:
        if np.hypot(15 - vehicle_1.x, -2 - vehicle_1.y) < 1:
            break

        acc, omega = planner.planning(vehicle_1, [15, -2])
        vehicle_1.update(acc, omega, 0.5)

        plt.cla()
        env.draw_env()
        vehicle_0.draw_vehicle()
        vehicle_1.draw_vehicle()
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.gca().set_aspect('equal')
        plt.pause(0.5)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    run()

