import logging.handlers
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

from utils import Node, State, kinematic_propagate
from env import EnvCrossroads
from vehicle_base import VehicleBase
from vehicle import Vehicle, VehicleList
from planner import MonteCarloTreeSearch


LOG_LEVEL_DICT = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING,
                  3: logging.ERROR, 4: logging.CRITICAL}


def run(rounds_num:int, config_path:str, save_path:str, show_animation:bool, save_fig:bool) -> None:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        logging.info(f"Config parameters:\n{config}")

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
            Vehicle("vehicle_0", State(env.lanewidth / 2, init_y_0, np.pi / 2, init_v_0), 'b', config)
        # the straight vehicle
        vehicle_1 = \
            Vehicle("vehicle_1", State(-env.lanewidth / 2, init_y_1, -np.pi / 2, init_v_1), 'r', config)

        vehicle_0.set_level(1)
        vehicle_1.set_level(0)
        vehicle_0.set_target(State(-18, env.lanewidth / 2, math.pi))
        vehicle_1.set_target(State(-env.lanewidth / 2, -18, 1.5 * math.pi))

        vehicles = VehicleList([vehicle_0, vehicle_1])

        vehicle_0_history: List[State] = [vehicles[0].state]
        vehicle_1_history: List[State] = [vehicles[1].state]

        logging.info(f"\n================== Round {iter} ==================")
        for vehicle in vehicles:
            logging.info(f"{vehicle.name} >>> init_x: {vehicle.state.x:.2f}, "
                         f"init_y: {vehicle.state.y:.2f}, init_v: {vehicle.state.v:.2f}")

        cur_loop_count = 0
        round_start_time = time.time()
        while True:
            if vehicles.is_all_get_target:
                round_elapsed_time = time.time() - round_start_time
                logging.info(f"Round {iter} successed, "
                             f"simulation time: {cur_loop_count * config['delta_t']} s"
                             f", actual timecost: {round_elapsed_time:.3f} s")
                succeed_count += 1
                break

            if vehicles.is_any_collision or cur_loop_count > max_per_iters:
                round_elapsed_time = time.time() - round_start_time
                logging.info(f"Round {iter} failed, "
                             f"simulation time: {cur_loop_count * config['delta_t']} s"
                             f", actual timecost: {round_elapsed_time:.3f} s")
                break

            future_list: List[Future] = []
            start_time = time.time()

            for vehicle in vehicles:
                future = executor.submit(vehicle.excute, vehicles.exclude(vehicle)[0])
                future_list.append(future)

            for vehicle, future in zip(vehicles, future_list):
                vehicle.cur_action, vehicle.excepted_traj = future.result()
                if not vehicle.is_get_target:
                    vehicle.state = \
                        kinematic_propagate(vehicle.state, vehicle.cur_action.value, config['delta_t'])
 
            vehicle_0_history.append(vehicles[0].state)
            vehicle_1_history.append(vehicles[1].state)

            elapsed_time = time.time() - start_time
            logging.debug(f"single step cost {elapsed_time:.6f} second")

            if show_animation:
                plt.cla()
                env.draw_env()
                for vehicle in vehicles:
                    excepted_traj = vehicle.excepted_traj.to_list()
                    vehicle.draw_vehicle()
                    plt.plot(vehicle.target.x, vehicle.target.y, marker='x', color=vehicle.color)
                    plt.plot(excepted_traj[0][1:], excepted_traj[1][1:], color=vehicle.color, linewidth=1)
                plt.text(10, -15, f"v = {vehicles[0].state.v:.2f} m/s", color='blue')
                plt.text(10,  15, f"v = {vehicles[1].state.v:.2f} m/s", color='red')
                action_text = "GOAL !" if vehicles[0].is_get_target else vehicles[0].cur_action.name
                plt.text(10, -18, action_text, fontsize=10, color='blue')
                action_text = "GOAL !" if vehicles[1].is_get_target else vehicles[1].cur_action.name
                plt.text(10, 12, action_text, fontsize=10, color='red')
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

    logging.info("\n=========================================")
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
        args.output_path = os.path.dirname(os.path.dirname(current_file_path))

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

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(project_path, 'config', args.config)

    run(args.rounds, config_file_path, result_save_path, args.show, args.save_fig)
