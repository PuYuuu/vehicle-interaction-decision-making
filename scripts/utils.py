import numpy as np
from typing import List
from typing import Optional
from enum import Enum
import math
import random

class Action(Enum):
    """Enum of action sets for vehicle."""

    MAINTAIN = [0, 0]              # 'maintain'
    TURNLEFT = [0, math.pi / 6]    # 'turn left'
    TURNRIGHT = [0, -math.pi / 6]  # 'turn right'
    ACCELERATE = [2.5, 0]          # 'accelerate'
    DECELERATE = [-2.5, 0]         # 'decelerate'
    BRAKE = [-5, 0]                # 'brake'

ActionList = [Action.MAINTAIN, Action.TURNLEFT, Action.TURNRIGHT,
              Action.ACCELERATE, Action.DECELERATE, Action.BRAKE]

class Node:
    MAX_LEVEL: int = 6

    def __init__(self, x = 0, y = 0, yaw = 0, v = 0, level = 0,
                 p: Optional["Node"] = None, action: Optional[Action] = None):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

        self.value: float = 0
        self.reward: float = 0
        self.visits: int = 0
        self.action: Action = action
        self.parent: Node = p
        self.cur_level: int = level

        self.children: List[Node] = []
        self.actions: List[Action] = []

    @property
    def is_terminal(self) -> bool:
        return self.cur_level >= Node.MAX_LEVEL

    @property
    def is_fully_expanded(self) -> bool:
        if len(self.children) == len(Action):
            return True

        return False

    def add_child(self, next_action: Action, wb: float, delta_t: float) -> "Node":
        x, y, yaw, v = kinematic_propagate(self, next_action.value, wb, delta_t)
        node = Node(x, y, yaw, v, self.cur_level + 1, self, next_action)
        node.actions = self.actions + [next_action]
        self.children.append(node)

        return node

    def next_node(self, wb: float, delta_t: float):
        next_action = random.choice(ActionList)
        x, y, yaw, v = kinematic_propagate(self, next_action.value, wb, delta_t)
        node = Node(x, y, yaw, v, self.cur_level + 1, None, next_action)

        return node

    def __repr__(self):
        return f"children: {len(self.children)}, visits: {self.visits}, reward: {self.reward}, actions: {self.actions}"

def has_overlap(box2d_0, box2d_1) -> bool:
    total_sides = []
    for i in range(1, len(box2d_0[0])):
        vec_x = box2d_0[0][i] - box2d_0[0][i - 1]
        vec_y = box2d_0[1][i] - box2d_0[1][i - 1]
        total_sides.append([vec_x, vec_y])
    for i in range(1, len(box2d_1[0])):
        vec_x = box2d_1[0][i] - box2d_1[0][i - 1]
        vec_y = box2d_1[1][i] - box2d_1[1][i - 1]
        total_sides.append([vec_x, vec_y])

    for i in range(len(total_sides)):
        separating_axis = [-total_sides[i][1], total_sides[i][0]]

        vehicle_min = np.inf
        vehicle_max = -np.inf
        for j in range(0, len(box2d_0[0])):
            project = separating_axis[0] * box2d_0[0][j] + separating_axis[1] * box2d_0[1][j]
            vehicle_min = min(vehicle_min, project)
            vehicle_max = max(vehicle_max, project)

        box2d_min = np.inf
        box2d_max = -np.inf
        for j in range(0, len(box2d_1[0])):
            project = separating_axis[0] * box2d_1[0][j] + separating_axis[1] * box2d_1[1][j]
            box2d_min = min(box2d_min, project)
            box2d_max = max(box2d_max, project)

        if vehicle_min > box2d_max or box2d_min > vehicle_max:
            return False

    return True


def kinematic_propagate(node: Node, act: List[float], wheel_base: float, dt: float):
    acc, delta = act[0], act[1]

    omega = (np.tan(delta) / wheel_base) * node.v
    x = node.x + node.v * np.cos(node.yaw) * dt
    y = node.y + node.v * np.sin(node.yaw) * dt
    v = node.v + acc * dt
    yaw = node.yaw + omega * dt

    if yaw > 2 * np.pi:
        yaw -= 2 * np.pi
    if yaw < 0:
        yaw += 2 * np.pi

    if v > 20:
        v = 20
    elif v < -20:
        v = -20

    return x, y, yaw, v
