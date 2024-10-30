'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-05-12 22:33:57
LastEditTime: 2024-10-31 01:00:49
FilePath: /vehicle-interaction-decision-making/scripts/env.py
Copyright 2024 puyu, All Rights Reserved.
'''

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
        for i in range(len(self.rect)):
            plt.fill(*self.rect[i], color='gray', alpha=0.5)

        for i in range(len(self.rect)):
            plt.plot(*self.rect[i], color='k', linewidth=2)

        for i in range(len(self.laneline)):
            plt.plot(*self.laneline[i], linestyle='--', color='orange', linewidth=2)
