'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-05-27 22:23:54
LastEditTime: 2024-10-31 01:00:59
FilePath: /vehicle-interaction-decision-making/scripts/imshow.py
Copyright 2024 puyu, All Rights Reserved.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

def imshow(image: np.ndarray, state: np.ndarray, vehicle_para: np.ndarray) -> None:
    x, y, theta = state[0], state[1], state[2]
    length, width = vehicle_para[0], vehicle_para[1]

    transform_data = Affine2D().rotate_deg_around(x, y, theta / np.pi * 180)
    transform_data += plt.gca().transData

    # 0.15 offset consider the length of the rearview mirror
    image_extent = [x - length / 2, x + length / 2,
                    y - width / 2 - 0.15, y + width / 2 + 0.15]
    plt.imshow(image, transform=transform_data,
               extent=image_extent, zorder=10.0, clip_on=True)

