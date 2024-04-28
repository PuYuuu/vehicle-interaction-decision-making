import os
import math
import time
import yaml
import random
import logging
import argparse
from datetime import datetime
from typing import List
from concurrent.futures import ProcessPoolExecutor, Future

import numpy as np
import matplotlib.pyplot as plt

from utils import Node, State, has_overlap, kinematic_propagate
from env import EnvCrossroads
from vehicle_base import VehicleBase
from vehicle import Vehicle
from planner import MonteCarloTreeSearch

LOG_LEVEL_DICT = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING,
                  3: logging.ERROR, 4: logging.CRITICAL}

def run(rounds_num:int, config_path:str, save_path:str, show_animation:bool, save_fig:bool) -> None:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print("Config parameters:\n", config)

    logging.info(f"rounds_num: {rounds_num}")
    env = EnvCrossroads(size = 25, lanewidth = 4.2)
    max_per_iters = 20 / config['delta_t']

    # initialize
    VehicleBase.initialize(env, 5, 2, 8, 2.4)
    MonteCarloTreeSearch.initialize(config)
    Node.initialize(config['max_step'], MonteCarloTreeSearch.calc_cur_value)

    succeed_count = 0
    executor = ProcessPoolExecutor(max_workers = 6)
    for iter in range(rounds_num):
        init_y_0 = random.uniform(-20, -12)
        init_y_1 = random.uniform(12, 20)
        init_v_0 = random.uniform(3, 5)
        init_v_1 = random.uniform(3, 5)

        # the turn left vehicle
        vehicle_0 = \
            Vehicle("vehicle_0", State(env.lanewidth / 2, init_y_0, np.pi / 2, init_v_0), 'blue', config)
        # the straight vehicle
        vehicle_1 = \
            Vehicle("vehicle_1", State(-env.lanewidth / 2, init_y_1, -np.pi / 2, init_v_1), 'red', config)

        vehicle_0.set_level(1)
        vehicle_1.set_level(0)
        vehicle_0.set_target(State(-18, env.lanewidth / 2, math.pi))
        vehicle_1.set_target(State(-env.lanewidth / 2, -18, 1.5 * math.pi))
        print(f"vehicle_1 tar: {vehicle_1.target.x}, {vehicle_1.target.y}")
        vehicle_0_history: List[State] = [vehicle_0.state]
        vehicle_1_history: List[State] = [vehicle_1.state]

        print(f"\n================== Round {iter} ==================")
        logging.info(f"Vehicle 0 >>> init_x: {vehicle_0.state.x:.2f}, "
                     f"init_y: {init_y_0:.2f}, init_v: {init_v_0:.2f}")
        logging.info(f"Vehicle 1 >>> init_x: {vehicle_1.state.x:.2f}, "
                     f"init_y: {init_y_1:.2f}, init_v: {init_v_1:.2f}")

        cur_loop_count = 0
        round_start_time = time.time()
        while True:
            if vehicle_0.is_get_target and vehicle_1.is_get_target:
                round_elapsed_time = time.time() - round_start_time
                logging.info(f"Round {iter} successed, "
                             f"simulation time: {cur_loop_count * config['delta_t']} s"
                             f", actual timecost: {round_elapsed_time:.3f} s")
                succeed_count += 1
                break

            if has_overlap(VehicleBase.get_box2d(vehicle_1.state), \
                           VehicleBase.get_box2d(vehicle_0.state)) or \
                           cur_loop_count > max_per_iters:
                round_elapsed_time = time.time() - round_start_time
                logging.info(f"Round {iter} failed, "
                             f"simulation time: {cur_loop_count * config['delta_t']} s"
                             f", actual timecost: {round_elapsed_time:.3f} s")
                break

            future_list: List[Future] = []
            start_time = time.time()

            future = executor.submit(vehicle_0.excute, vehicle_1)
            future_list.append(future)
            future = executor.submit(vehicle_1.excute, vehicle_0)
            future_list.append(future)

            act_0, excepted_traj_0 = future_list[0].result()
            if not vehicle_0.is_get_target:
                vehicle_0.state = kinematic_propagate(vehicle_0.state, act_0.value, config['delta_t'])
            act_1, excepted_traj_1 = future_list[1].result()
            if not vehicle_1.is_get_target:
                vehicle_1.state = kinematic_propagate(vehicle_1.state, act_1.value, config['delta_t'])
            vehicle_0_history.append(vehicle_0.state)
            vehicle_1_history.append(vehicle_1.state)

            elapsed_time = time.time() - start_time
            logging.debug(f"single step cost {elapsed_time:.6f} second")

            if show_animation:
                plt.cla()
                env.draw_env()
                vehicle_0.draw_vehicle()
                vehicle_1.draw_vehicle()
                plt.plot(vehicle_0.target.x, vehicle_0.target.y, "xb")
                plt.plot(vehicle_1.target.x, vehicle_1.target.y, "xr")
                plt.text(10, -15, f"v = {vehicle_0.state.v:.2f} m/s", color='blue')
                plt.text(10,  15, f"v = {vehicle_1.state.v:.2f} m/s", color='red')
                action_text = "GOAL !" if vehicle_0.is_get_target else act_0.name
                plt.text(10, -18, action_text, fontsize=10, color='blue')
                action_text = "GOAL !" if vehicle_1.is_get_target else act_1.name
                plt.text(10, 12, action_text, fontsize=10, color='red')
                plt.plot([traj[0] for traj in excepted_traj_0[1:]],
                         [traj[1] for traj in excepted_traj_0[1:]], color='blue', linewidth=1)
                plt.plot([traj[0] for traj in excepted_traj_1[1:]],
                         [traj[1] for traj in excepted_traj_1[1:]], color='red', linewidth=1)
                plt.xlim(-25, 25)
                plt.ylim(-25, 25)
                plt.gca().set_aspect('equal')
                plt.pause(0.01)
            cur_loop_count += 1

        plt.cla()
        env.draw_env()
        for history in vehicle_0_history:
            tmp = Vehicle("tmp", history, "blue", config)
            tmp.draw_vehicle(True)
        for history in vehicle_1_history:
            tmp = Vehicle("tmp", history, "red", config)
            tmp.draw_vehicle(True)
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.gca().set_aspect('equal')
        if save_fig:
            plt.savefig(os.path.join(save_path, f"round_{iter}.svg"), format='svg', dpi=600)

    print("\n=========================================")
    logging.info(f"Experiment success {succeed_count}/{rounds_num}"
                 f"({100*succeed_count/rounds_num:.2f}%) rounds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--rounds', '-r', type=int, default=5, help='')
    parser.add_argument('--output_path', '-o', type=str, default=None, help='')
    parser.add_argument('--log_level', '-l', type=int, default=1,
                        help=f"0:logging.DEBUG\t1:logging.INFO\t"
                             f"2:logging.WARNING\t3:logging.ERROR\t"
                             f"4:logging.CRITICAL\t")
    parser.add_argument('--config', '-c', type=str, default='default.yaml', help='')
    parser.add_argument('--show', action='store_true', default=False, help='')
    parser.add_argument('--save_fig', action='store_true', default=False, help='')
    args = parser.parse_args()

    if args.output_path is None:
        current_file_path = os.path.abspath(__file__)
        args.output_path = os.path.dirname(current_file_path)

    logging.basicConfig(level=LOG_LEVEL_DICT[args.log_level],
                        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    result_save_path = os.path.join(args.output_path, "logs", formatted_time)
    if args.save_fig:
        os.makedirs(result_save_path, exist_ok=True)
        logging.info(f"Experiment results save at \"{result_save_path}\"")

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(project_path, 'config', args.config)

    run(args.rounds, config_file_path, result_save_path, args.show, args.save_fig)
