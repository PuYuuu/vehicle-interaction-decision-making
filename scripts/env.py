import math
import numpy as np
import matplotlib.pyplot as plt

class EnvCrossroads:
    def __init__(self, size = 25, lanewidth = 4):
        self.map_size = 25
        self.lanewidth = 4

        self.rect = [
            [[-size, -size, -2*lanewidth, -lanewidth, -lanewidth, -size],
                [-size, -lanewidth, -lanewidth, -2*lanewidth, -size, -size]],
            [[size, size, 2*lanewidth, lanewidth, lanewidth, size],
                [-size, -lanewidth, -lanewidth, -2*lanewidth, -size, -size]],
            [[size, size, 2*lanewidth, lanewidth, lanewidth, size],
                [size, lanewidth, lanewidth, 2*lanewidth, size, size]],
            [[-size, -size, -2*lanewidth, -lanewidth, -lanewidth, -size],
                [size, lanewidth, lanewidth, 2*lanewidth, size, size]]
        ]

        self.laneline = [
            [[0, 0], [-size, -2*lanewidth]],
            [[0, 0], [size, 2*lanewidth]],
            [[-size, -2*lanewidth], [0, 0]],
            [[size, 2*lanewidth], [0, 0]]
        ]


    def draw_env(self):
        plt.fill(*self.rect[0], color='gray', alpha=0.5)
        plt.fill(*self.rect[1], color='gray', alpha=0.5)
        plt.fill(*self.rect[2], color='gray', alpha=0.5)
        plt.fill(*self.rect[3], color='gray', alpha=0.5)
        plt.plot(*self.rect[0], color='k', linewidth=2)
        plt.plot(*self.rect[1], color='k', linewidth=2)
        plt.plot(*self.rect[2], color='k', linewidth=2)
        plt.plot(*self.rect[3], color='k', linewidth=2)

        plt.plot(*self.laneline[0], linestyle='--', color='orange', linewidth=2)
        plt.plot(*self.laneline[1], linestyle='--', color='orange', linewidth=2)
        plt.plot(*self.laneline[2], linestyle='--', color='orange', linewidth=2)
        plt.plot(*self.laneline[3], linestyle='--', color='orange', linewidth=2)


class Vehicle:
    def __init__(self, x, y, yaw, length = 5, width = 2, color = 'k'):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = 0
        self.length = length
        self.width = width
        self.color = color


    def update(self, acc, omega, dt):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.v += acc * dt
        self.yaw += omega * dt


    def draw_vehicle(self):
        vehicle = np.array(
            [[-self.length/2, self.length/2, self.length/2, -self.length/2, -self.length/2],
            [self.width/2, self.width/2, -self.width/2, -self.width/2, self.width/2]]
        )
        head = np.array(
            [[0.3 * self.length, 0.3 * self.length], [self.width/2, -self.width/2]]
        )

        rot1 = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                     [np.sin(self.yaw), np.cos(self.yaw)]])

        vehicle = np.dot(rot1, vehicle)
        head = np.dot(rot1, head)

        vehicle += np.array([[self.x], [self.y]])
        head += np.array([[self.x], [self.y]])

        plt.plot(vehicle[0, :], vehicle[1, :], self.color)
        plt.plot(head[0, :], head[1, :], self.color)
