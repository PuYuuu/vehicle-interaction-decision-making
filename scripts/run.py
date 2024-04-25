import os
import math
import time
import random
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import utils
from env import EnvCrossroads
from vehicle import Vehicle

DT = 0.5


def run(rounds_num:int, save_path:str, show_animation:bool) -> None:
    print(f"rounds_num: {rounds_num}")
    env = EnvCrossroads(size=25, lanewidth=4.2)
    max_per_iters = 25 / DT

    succeed_count = 0
    for iter in range(rounds_num):
        init_y_0 = random.uniform(-20, -12)
        init_y_1 = random.uniform(12, 20)
        init_v_0 = random.uniform(3, 5)
        init_v_1 = random.uniform(3, 5)

        # the turn left vehicle
        vehicle_0 = Vehicle("vehicle_0", env.lanewidth / 2, init_y_0, np.pi / 2, env, DT, 'blue')
        # the straight vehicle
        vehicle_1 = Vehicle("vehicle_1", -env.lanewidth / 2, init_y_1, -np.pi / 2, env, DT, 'red')

        vehicle_0.set_level(1)
        vehicle_1.set_level(0)
        vehicle_0.set_target([-18, env.lanewidth / 2, math.pi])
        vehicle_1.set_target([-env.lanewidth / 2, -18, 1.5 * math.pi])
        vehicle_0.v = init_v_0
        vehicle_1.v = init_v_1

        vehicle_0_history = [[env.lanewidth / 2, init_y_0, np.pi / 2]]
        vehicle_1_history = [[-env.lanewidth / 2, init_y_1, -np.pi / 2]]

        print("\n================== Round {} ==================".format(iter))
        print(f"Vehicle 0 >>> init_x: {vehicle_0.x:.2f}, init_y: {init_y_0:.2f}, init_v: {init_v_0:.2f}")
        print(f"Vehicle 1 >>> init_x: {vehicle_1.x:.2f}, init_y: {init_y_1:.2f}, init_v: {init_v_1:.2f}")

        cur_loop_count = 0
        while True:
            if vehicle_0.is_get_target and vehicle_1.is_get_target:
                print("success !")
                succeed_count += 1
                break

            if utils.has_overlap(vehicle_1.get_box2d(), vehicle_0.get_box2d()) or \
                cur_loop_count > max_per_iters:
                print("failed, exit...")
                break

            start_time = time.time()
            if not vehicle_0.is_get_target:
                act_0, excepted_traj_0 = vehicle_0.excute(vehicle_1)
                vehicle_0_history.append([vehicle_0.x, vehicle_0.y, vehicle_0.yaw])
 
            if not vehicle_1.is_get_target:
                act_1, excepted_traj_1 = vehicle_1.excute(vehicle_0)
                vehicle_1_history.append([vehicle_1.x, vehicle_1.y, vehicle_1.yaw])
            elapsed_time = time.time() - start_time
            # print("cost time: {}".format(elapsed_time))

            if show_animation:
                plt.cla()
                env.draw_env()
                vehicle_0.draw_vehicle()
                vehicle_1.draw_vehicle()
                plt.plot(vehicle_0.target[0], vehicle_0.target[1], "xb")
                plt.plot(vehicle_1.target[0], vehicle_1.target[1], "xr")
                plt.text(10, -15, f"v = {vehicle_0.v:.2f} m/s", color='blue')
                plt.text(10,  15, f"v = {vehicle_1.v:.2f} m/s", color='red')
                plt.text(10, -18, act_0.name, fontsize=10, color='blue')
                plt.text(10,  12, act_1.name, fontsize=10, color='red')
                plt.plot([traj[0] for traj in excepted_traj_0[1:]],
                         [traj[1] for traj in excepted_traj_0[1:]], color='blue', linewidth=1)
                plt.plot([traj[0] for traj in excepted_traj_1[1:]],
                         [traj[1] for traj in excepted_traj_1[1:]], color='red', linewidth=1)
                plt.xlim(-25, 25)
                plt.ylim(-25, 25)
                plt.gca().set_aspect('equal')
                plt.pause(0.1)
            cur_loop_count += 1

        plt.cla()
        env.draw_env()
        for history in vehicle_0_history:
            tmp = Vehicle("tmp", history[0], history[1], history[2], env, DT, "blue")
            tmp.draw_vehicle(True)
        for history in vehicle_1_history:
            tmp = Vehicle("tmp", history[0], history[1], history[2], env, DT, "red")
            tmp.draw_vehicle(True)
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.gca().set_aspect('equal')
        plt.savefig(os.path.join(save_path, f"round_{iter}.svg"), format='svg', dpi=600)

    print("\n=========================================")
    print(f"Experiment success {succeed_count}/{rounds_num}({100*succeed_count/rounds_num:.2f}%) rounds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--rounds', '-r', type = int, default = 5, help='')
    parser.add_argument('--output_path', '-o', type = str, default = None, help='')
    parser.add_argument('--show', action='store_true', default=False, help = '')
    args = parser.parse_args()

    if args.output_path is None:
        current_file_path = os.path.abspath(__file__)
        args.output_path = os.path.dirname(current_file_path)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    result_save_path = os.path.join(args.output_path, "logs", formatted_time)
    os.makedirs(result_save_path, exist_ok=True)
    print(f"Experiment results save at \"{result_save_path}\"")

    run(args.rounds, result_save_path, args.show)
