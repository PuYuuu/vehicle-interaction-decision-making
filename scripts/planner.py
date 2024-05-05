import math
import random
import logging
import numpy as np
from typing import Tuple, List

import utils
from utils import Node, StateList
from vehicle_base import VehicleBase


class MonteCarloTreeSearch:
    EXPLORATE_RATE = 1 / ( 2 * math.sqrt(2.0))
    LAMDA = 0.9
    WEIGHT_AVOID = 10
    WEIGHT_SAFE = 0.2
    WEIGHT_OFFROAD = 2
    WEIGHT_VELOCITY = 10
    WEIGHT_DIRECTION = 1
    WEIGHT_DISTANCE = 0.1

    def __init__(self, ego: VehicleBase, other: VehicleBase,
                 other_traj: StateList, cfg: dict = {}):
        self.ego_vehicle: VehicleBase = ego
        self.other_vehicle: VehicleBase = other
        self.other_predict_traj: StateList = other_traj

        self.computation_budget = cfg['computation_budget']
        self.dt = cfg['delta_t']


    def excute(self, root: Node) -> Node:
        for _ in range(self.computation_budget):
            # 1. Find the best node to expand
            expand_node = self.tree_policy(root)
            # 2. Random run to add node and get reward
            reward = self.default_policy(expand_node)
            # 3. Update all passing nodes with reward
            self.backup(expand_node, reward)

        return self.get_best_child(root, 0)

    def tree_policy(self, node: Node) -> Node:
        while node.is_terminal == False:
            if len(node.children) == 0:
                return self.expand(node)
            elif random.uniform(0, 1) < .5:
                node = self.get_best_child(node, MonteCarloTreeSearch.EXPLORATE_RATE)
            else:
                if node.is_fully_expanded == False:    
                    return self.expand(node)
                else:
                    node = self.get_best_child(node, MonteCarloTreeSearch.EXPLORATE_RATE)

        return node

    def default_policy(self, node: Node) -> float:
        while node.is_terminal == False:
            cur_other_state = self.other_predict_traj[node.cur_level + 1]
            next_node = node.next_node(self.dt, [cur_other_state])
            node = next_node

        return node.value

    def backup(self, node: Node, r: float) -> None:
        while node != None:
            node.visits += 1
            node.reward += r
            node = node.parent

    def expand(self, node: Node) -> Node:
        tried_actions = [c.action for c in node.children]
        next_action = random.choice(utils.ActionList)
        while node.is_terminal == False and next_action in tried_actions:
            next_action = random.choice(utils.ActionList)
        cur_other_state = self.other_predict_traj[node.cur_level + 1]
        node.add_child(next_action, self.dt, [cur_other_state])

        return node.children[-1]

    def get_best_child(self, node: Node, scalar: float) -> Node:
        bestscore = -math.inf
        bestchildren = []
        for child in node.children:
            exploit = child.reward / child.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
            score = exploit + scalar * explore
            if score == bestscore:
                bestchildren.append(child)
            if score > bestscore:
                bestchildren = [child]
                bestscore = score
        if len(bestchildren) == 0:
            logging.debug("No best child found, probably fatal !")
            return node

        return random.choice(bestchildren)

    @staticmethod
    def calc_cur_value(node: Node, last_node_value: float) -> float:
        x, y, yaw = node.state.x, node.state.y, node.state.yaw
        step = node.cur_level
        ego_box2d = VehicleBase.get_box2d(node.state)
        ego_safezone = VehicleBase.get_safezone(node.state)

        avoid = 0
        safe = 0
        for cur_other_state in node.other_agent_state:
            if utils.has_overlap(ego_box2d, VehicleBase.get_box2d(cur_other_state)):
                avoid = -1
            if utils.has_overlap(ego_safezone, VehicleBase.get_safezone(cur_other_state)):
                safe = -1

        offroad = 0
        for rect in VehicleBase.env.rect:
            if utils.has_overlap(ego_box2d, rect):
                offroad = -1
                break

        direction = 0
        if MonteCarloTreeSearch.is_opposite_direction(node.state, ego_box2d):
            direction = -1

        velocity = 0
        if velocity < 0:
            velocity = -1

        distance = -(abs(x - node.goal_pos.x) + abs(y - node.goal_pos.y) + \
                     1.5 * abs(yaw - node.goal_pos.yaw))

        cur_reward = MonteCarloTreeSearch.WEIGHT_AVOID * avoid + \
                     MonteCarloTreeSearch.WEIGHT_SAFE * safe + \
                     MonteCarloTreeSearch.WEIGHT_OFFROAD * offroad + \
                     MonteCarloTreeSearch.WEIGHT_DISTANCE * distance + \
                     MonteCarloTreeSearch.WEIGHT_DIRECTION * direction + \
                     MonteCarloTreeSearch.WEIGHT_VELOCITY * velocity

        total_reward = last_node_value + MonteCarloTreeSearch.LAMDA ** (step - 1) * cur_reward
        node.value = total_reward

        return total_reward

    @staticmethod
    def is_opposite_direction(pos: utils.State, ego_box2d = None) -> bool:
        x, y, yaw = pos.x, pos.y, pos.yaw
        if ego_box2d is None:
            ego_box2d = VehicleBase.get_box2d(pos)

        for laneline in VehicleBase.env.laneline:
            if utils.has_overlap(ego_box2d, laneline):
                return True

        lanewidth = VehicleBase.env.lanewidth

        # down lane
        if x > -lanewidth and x < 0 and (y < -lanewidth or y > lanewidth):
            if yaw > 0 and yaw < np.pi:
                return True
        # up lane
        elif x > 0 and x < lanewidth and (y < -lanewidth or y > lanewidth):
            if not (yaw > 0 and yaw < np.pi):
                return True
        # right lane
        elif y > -lanewidth and y < 0 and (x < -lanewidth or x > lanewidth):
            if yaw > 0.5 * np.pi and yaw < 1.5 * np.pi:
                return True
        # left lane
        elif y > 0 and y < lanewidth and (x < -lanewidth or x > lanewidth):
            if not (yaw > 0.5 * np.pi and yaw < 1.5 * np.pi):
                return True

        return False

    @staticmethod
    def initialize(cfg: dict = {}) -> None:
        MonteCarloTreeSearch.LAMDA = cfg['lamda']
        MonteCarloTreeSearch.WEIGHT_AVOID = cfg['weight_avoid']
        MonteCarloTreeSearch.WEIGHT_SAFE = cfg['weight_safe']
        MonteCarloTreeSearch.WEIGHT_OFFROAD = cfg['weight_offroad']
        MonteCarloTreeSearch.WEIGHT_VELOCITY = cfg['weight_velocity']
        MonteCarloTreeSearch.WEIGHT_DIRECTION = cfg['weight_direction']
        MonteCarloTreeSearch.WEIGHT_DISTANCE = cfg['weight_distance']


class KLevelPlanner:
    def __init__(self, cfg: dict = {}):
        self.steps = cfg['max_step']
        self.dt = cfg['delta_t']
        self.config = cfg

    def planning(self, ego: VehicleBase, other: VehicleBase) -> Tuple[utils.Action, StateList]:
        other_prediction = self.get_prediction(ego, other)
        actions, traj = self.forward_simulate(ego, other, other_prediction)

        return actions[0], traj

    def forward_simulate(self, ego: VehicleBase, other: VehicleBase, traj: StateList) -> Tuple[List[utils.Action], StateList]:
        mcts = MonteCarloTreeSearch(ego, other, traj, self.config)
        current_node = Node(state = ego.state, goal = ego.target)
        current_node = mcts.excute(current_node)
        for _ in range(Node.MAX_LEVEL - 1):
            current_node = mcts.get_best_child(current_node, 0)

        actions = current_node.actions
        state_list = StateList()
        while current_node != None:
            state_list.append(current_node.state)
            current_node = current_node.parent
        expected_traj = state_list.reverse()

        if len(expected_traj) < self.steps + 1:
            logging.debug(f"The max level of the node is not enough({len(expected_traj)}),"
                          f"using the last value to complete it.")
            expected_traj.expand(self.steps + 1)

        return actions, expected_traj

    def get_prediction(self, ego: VehicleBase, other: VehicleBase) -> StateList:
        if ego.level == 0 or other.is_get_target:
            return StateList([other.state] * (self.steps + 1))
        elif ego.level == 1:
            other_prediction_ego = StateList([ego.state] * (self.steps + 1))
            other_act, other_traj = self.forward_simulate(other, ego, other_prediction_ego)
            return other_traj
        elif ego.level == 2:
            static_traj = StateList([other.state] * (self.steps + 1))
            _, ego_l0_traj = self.forward_simulate(ego, other, static_traj)
            _, other_l1_traj = self.forward_simulate(other, ego, ego_l0_traj)
            return other_l1_traj
