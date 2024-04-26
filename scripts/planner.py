import math
import random
import logging
import numpy as np

import utils
from utils import Node
from env import EnvCrossroads
from vehicle_base import VehicleBase

LAMDA = 0.9
WEIGHT_AVOID = 10
WEIGHT_SAFE = 0.2
WEIGHT_OFFROAD = 2
WEIGHT_VELOCITY = 10
WEIGHT_DIRECTION = 1
WEIGHT_DISTANCE = 0.1


class MonteCarloTreeSearch:
    EXPLORATE_RATE = 1 / ( 2 * math.sqrt(2.0))

    def __init__(self, ego: VehicleBase, other: VehicleBase, env: EnvCrossroads,
                 other_traj, budget: int = 1000, dt: float = 0.1):
        self.ego_vehicle = ego
        self.other_vehicle = other
        self.env = env
        self.other_predict_traj = other_traj
        self.computation_budget = budget
        self.dt = dt

    def excute(self, root: Node):

        for _ in range(self.computation_budget):
            # 1. Find the best node to expand
            expand_node = self.tree_policy(root)
            # 2. Random run to add node and get reward
            reward = self.default_policy(expand_node)
            # 3. Update all passing nodes with reward
            self.backup(expand_node, reward)

        return self.get_best_child(root, 0)

    def tree_policy(self, node: Node):
        while node.is_terminal == False:
            if len(node.children) == 0:
                return self.expand(node)
            elif random.uniform(0, 1) < .5:
                node = self.get_best_child(node, self.EXPLORATE_RATE)
            else:
                if node.is_fully_expanded == False:    
                    return self.expand(node)
                else:
                    node = self.get_best_child(node, self.EXPLORATE_RATE)

        return node

    def default_policy(self, node: Node):
        while node.is_terminal == False:
            next_node = node.next_node(self.dt)
            self.calc_cur_value(next_node, node.value)
            node = next_node

        return node.value

    def backup(self, node: Node, r: float):
        while node != None:
            node.visits += 1
            node.reward += r
            node = node.parent

    def expand(self, node: utils.Node) -> utils.Node:
        tried_actions = [c.action for c in node.children]
        next_action = random.choice(utils.ActionList)
        while node.is_terminal == False and next_action in tried_actions:
            next_action = random.choice(utils.ActionList)
        node.add_child(next_action, self.dt)
        self.calc_cur_value(node.children[-1], node.value)

        return node.children[-1]

    def get_best_child(self, node: Node, scalar: float) -> utils.Node:
        bestscore = -math.inf
        bestchildren = []
        for c in node.children:
            exploit = c.reward / c.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / c.visits)
            score = exploit + scalar * explore
            if score == bestscore:
                bestchildren.append(c)
            if score > bestscore:
                bestchildren = [c]
                bestscore = score
        if len(bestchildren) == 0:
            logging.debug("No best child found, probably fatal !")
            return node

        return random.choice(bestchildren)

    def calc_cur_value(self, node: Node, last_node_value: float) -> float:
        x, y, yaw = node.x, node.y, node.yaw
        step = node.cur_level
        ego_box2d = self.ego_vehicle.get_box2d([x, y])
        ego_safezone = self.ego_vehicle.get_safezone([x, y])

        avoid = 0
        if utils.has_overlap(ego_box2d, self.other_vehicle.get_box2d(self.other_predict_traj[step])):
            avoid = -1
        safe = 0
        if utils.has_overlap(ego_safezone, self.other_vehicle.get_safezone(self.other_predict_traj[step])):
            safe = -1
        offroad = 0
        for rect in self.env.rect:
            if utils.has_overlap(ego_box2d, rect):
                offroad = -1
                break

        direction = 0
        if self.is_opposite_direction(x, y, yaw, ego_box2d):
            direction = -1

        velocity = 0
        if velocity < 0:
            velocity = -1

        distance = -(abs(x - self.ego_vehicle.target[0]) + abs(y - self.ego_vehicle.target[1]) + \
                      1.5 * abs(yaw - self.ego_vehicle.target[2]))

        cur_reward = WEIGHT_AVOID * avoid +  WEIGHT_SAFE * safe + \
                     WEIGHT_OFFROAD * offroad + WEIGHT_DISTANCE * distance + \
                     WEIGHT_DIRECTION * direction + WEIGHT_VELOCITY * velocity

        total_reward = last_node_value + LAMDA ** (step - 1) * cur_reward
        node.value = total_reward

        return total_reward

    def is_opposite_direction(self, x, y, yaw, ego_box2d):
        for laneline in self.env.laneline:
            if utils.has_overlap(ego_box2d, laneline):
                return True

        lanewidth = self.env.lanewidth

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


class KLevelPlanner:
    def __init__(self, env: EnvCrossroads, steps = 6, dt = 0.1):
        self.env = env
        self.steps = steps
        self.dt = dt


    def planning(self, ego:VehicleBase, other:VehicleBase):
        other_prediction = self.get_prediction(ego, other)
        actions, traj = self.forward_simulate(ego, other, other_prediction)

        return actions[0], traj


    def forward_simulate(self, ego: VehicleBase, other: VehicleBase, traj):
        mcts = MonteCarloTreeSearch(ego, other, self.env, traj, 10000, self.dt)
        current_node = Node(ego.x, ego.y, ego.yaw, ego.v)
        current_node = mcts.excute(current_node)
        for _ in range(Node.MAX_LEVEL - 1):
            current_node = mcts.get_best_child(current_node, 0)

        actions = [act for act in current_node.actions]
        pos_list = []
        while current_node != None:
            pos_list.append([current_node.x, current_node.y])
            current_node = current_node.parent
        expected_traj = pos_list[::-1]

        if len(expected_traj) < self.steps + 1:
            last_expected_pos = expected_traj[-1]
            logging.debug(f"The max level of the node is not enough{len(expected_traj)},"
                          f"using the last value to complete it.")
            for _ in range(self.steps + 1 - len(expected_traj)):
                expected_traj.append(last_expected_pos)

        return actions, expected_traj


    def get_prediction(self, ego: VehicleBase, other: VehicleBase):
        if ego.level == 0 or other.is_get_target:
            return [[other.x, other.y]] * (self.steps + 1)
        elif ego.level == 1:
            other_prediction_ego = [[ego.x, ego.y]] * (self.steps + 1)
            other_act, other_traj = self.forward_simulate(other, ego, other_prediction_ego)
            return other_traj
        elif ego.level == 2:
            static_traj = [[other.x, other.y]] * (self.steps + 1)
            _, ego_l0_traj = self.forward_simulate(ego, other, static_traj)
            _, other_l1_traj = self.forward_simulate(other, ego, ego_l0_traj)
            return other_l1_traj
