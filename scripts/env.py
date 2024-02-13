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
