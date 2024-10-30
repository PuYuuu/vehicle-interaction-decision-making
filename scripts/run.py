'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-05-28 23:07:35
LastEditTime: 2024-10-31 01:01:18
FilePath: /vehicle-interaction-decision-making/scripts/run.py
Copyright 2024 puyu, All Rights Reserved.
'''

import os
import time
import yaml
import logging
import argparse
from datetime import datetime
from typing import List
from concurrent.futures import ProcessPoolExecutor, Future

import matplotlib.pyplot as plt

from utils import Node, kinematic_propagate
from env import EnvCrossroads
from vehicle_base import VehicleBase
from vehicle import Vehicle, VehicleList
from planner import MonteCarloTreeSearch


LOG_LEVEL_DICT = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING,
                  "error": logging.ERROR, "critical": logging.CRITICAL}


def run(rounds_num:int, config_path:str, save_path:str, no_animation:bool, save_fig:bool) -> None:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        logging.info(f"Config parameters:\n{config}")

    logging.info(f"rounds_num: {rounds_num}")
    map_size = config["map_size"]
    lane_width = config["lane_width"]
    env = EnvCrossroads(map_size, lane_width)
    delta_t = config['delta_t']
    max_simulation_time = config['max_simulation_time']
    vehicle_draw_style = config['vehicle_display_style']

    # initialize
    VehicleBase.initialize(env, 5, 2, 8, 2.4)
    MonteCarloTreeSearch.initialize(config)
    Node.initialize(config['max_step'], MonteCarloTreeSearch.calc_cur_value)
    
    vehicles = VehicleList()
    for vehicle_name in config["vehicle_list"]:
        vehicle = Vehicle(vehicle_name, config)
        vehicles.append(vehicle)

    succeed_count = 0
    executor = ProcessPoolExecutor(max_workers = 6)
    for iter in range(rounds_num):
        vehicles.reset()

        logging.info(f"\n================== Round {iter} ==================")
        for vehicle in vehicles:
            logging.info(f"{vehicle.name} >>> init_x: {vehicle.state.x:.2f}, "
                         f"init_y: {vehicle.state.y:.2f}, init_v: {vehicle.state.v:.2f}")

        timestamp = 0.0
        round_start_time = time.time()
        while True:
            if vehicles.is_all_get_target:
                round_elapsed_time = time.time() - round_start_time
                logging.info(f"Round {iter} successed, "
                             f"simulation time: {timestamp} s"
                             f", actual timecost: {round_elapsed_time:.3f} s")
                succeed_count += 1
                break

            if vehicles.is_any_collision or timestamp > max_simulation_time:
                round_elapsed_time = time.time() - round_start_time
                logging.info(f"Round {iter} failed, "
                             f"simulation time: {timestamp} s"
                             f", actual timecost: {round_elapsed_time:.3f} s")
                break

            future_list: List[Future] = []
            start_time = time.time()

            for vehicle in vehicles:
                future = executor.submit(vehicle.excute, vehicles.exclude(vehicle))
                future_list.append(future)

            for vehicle, future in zip(vehicles, future_list):
                vehicle.cur_action, vehicle.excepted_traj = future.result()
                if not vehicle.is_get_target:
                    vehicle.state = \
                        kinematic_propagate(vehicle.state, vehicle.cur_action.value, delta_t)
                    vehicle.footprint.append(vehicle.state)

            elapsed_time = time.time() - start_time
            logging.debug(f"simulation time {timestamp:.3f} step cost {elapsed_time:.6f} second")

            if not no_animation:
                plt.cla()
                env.draw_env()
                for vehicle in vehicles:
                    excepted_traj = vehicle.excepted_traj.to_list()
                    vehicle.draw_vehicle(vehicle_draw_style)
                    plt.plot(vehicle.target.x, vehicle.target.y, marker='x', color=vehicle.color)
                    plt.plot(excepted_traj[0], excepted_traj[1], color=vehicle.color, linewidth=1)
                    plt.text(vehicle.vis_text_pos.x, vehicle.vis_text_pos.y + 3, f"level {vehicle.level}", color=vehicle.color)
                    plt.text(vehicle.vis_text_pos.x, vehicle.vis_text_pos.y,
                             f"v = {vehicle.state.v:.2f} m/s", color=vehicle.color)
                    action_text = "GOAL !" if vehicle.is_get_target else vehicle.cur_action.name
                    plt.text(vehicle.vis_text_pos.x, vehicle.vis_text_pos.y - 3, action_text, color=vehicle.color)
                plt.xlim(-map_size, map_size)
                plt.ylim(-map_size, map_size)
                plt.title(f"Round {iter + 1} / {rounds_num}")
                plt.gca().set_aspect('equal')
                plt.pause(0.01)
            timestamp += delta_t

        plt.cla()
        env.draw_env()
        for vehicle in vehicles:
            for state in vehicle.footprint:
                vehicle.state = state
                vehicle.draw_vehicle(vehicle_draw_style, True)
            plt.text(vehicle.vis_text_pos.x, vehicle.vis_text_pos.y + 3,
                     f"level {vehicle.level}", color=vehicle.color)
        plt.xlim(-map_size, map_size)
        plt.ylim(-map_size, map_size)
        plt.title(f"Round {iter + 1} / {rounds_num}")
        plt.gca().set_aspect('equal')
        plt.pause(1)
        if save_fig:
            plt.savefig(os.path.join(save_path, f"round_{iter}.svg"), format='svg', dpi=600)

    logging.info("\n=========================================")
    logging.info(f"Experiment success {succeed_count}/{rounds_num}"
                 f"({100*succeed_count/rounds_num:.2f}%) rounds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--rounds', '-r', type=int, default=5, help='')
    parser.add_argument('--output_path', '-o', type=str, default=None, help='')
    parser.add_argument('--log_level', '-l', type=str, default="info",
                        help=f"debug:logging.DEBUG\tinfo:logging.INFO\t"
                             f"warning:logging.WARNING\terror:logging.ERROR\t"
                             f"critical:logging.CRITICAL\t")
    parser.add_argument('--config', '-c', type=str, default=None, help='')
    parser.add_argument('--no_animation', action='store_true', default=False, help='')
    parser.add_argument('--save_fig', action='store_true', default=False, help='')
    args = parser.parse_args()

    current_file_path = os.path.abspath(__file__)
    if args.output_path is None:
        args.output_path = os.path.dirname(os.path.dirname(current_file_path))
    if args.config is None:
        config_file_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)),
                                        'config', 'unprotected_left_turn.yaml')
    else:
        config_file_path = args.config

    log_level = LOG_LEVEL_DICT[args.log_level]
    log_format = '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    result_save_path = os.path.join(args.output_path, "logs", formatted_time)
    if args.save_fig:
        os.makedirs(result_save_path, exist_ok=True)
        log_file_path = os.path.join(result_save_path, 'log')
        logging.basicConfig(level=log_level, format=log_format,
                            handlers=[logging.StreamHandler(), logging.FileHandler(filename=log_file_path)])
        logging.info(f"Experiment results save at \"{result_save_path}\"")
    else:
        logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    logging.info(f"log level : {args.log_level}")

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    run(args.rounds, config_file_path, result_save_path, args.no_animation, args.save_fig)
