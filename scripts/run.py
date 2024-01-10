import math
import numpy as np
import matplotlib.pyplot as plt

from env import *

def run():
    env = EnvCrossroads(size=25, lanewidth=4)
    vehicle_0 = Vehicle(2, -16, np.pi / 2, 5, 2, 'blue')
    vehicle_1 = Vehicle(-2, 16, -np.pi / 2, 5, 2, 'red')
    env.draw_env()
    vehicle_0.draw_vehicle()
    vehicle_1.draw_vehicle()

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    run()

