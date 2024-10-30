'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-04-27 16:17:27
LastEditTime: 2024-10-31 01:01:25
FilePath: /vehicle-interaction-decision-making/scripts/utils.py
Copyright 2024 puyu, All Rights Reserved.
'''

import math
import random
import numpy as np
from enum import Enum
from typing import List, Optional, Tuple


class Action(Enum):
    """Enum of action sets for vehicle."""
    MAINTAIN = [0, 0]              # maintain
    TURNLEFT = [0, math.pi / 4]    # turn left
    TURNRIGHT = [0, -math.pi / 4]  # turn right
    ACCELERATE = [2.5, 0]          # accelerate
    DECELERATE = [-2.5, 0]         # decelerate
    BRAKE = [-5, 0]                # brake

ActionList = [Action.MAINTAIN, Action.TURNLEFT, Action.TURNRIGHT,
              Action.ACCELERATE, Action.DECELERATE, Action.BRAKE]


class State:
    def __init__(self, x=0, y=0, yaw=0, v=0) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def to_list(self) -> List:
        return [self.x, self.y, self.yaw, self.v]


class StateList:
    def __init__(self, state_list = None) -> None:
        self.state_list: List[State] = state_list if state_list is not None else []

    def append(self, state: State) -> None:
        self.state_list.append(state)

    def reverse(self) -> 'StateList':
        self.state_list = self.state_list[::-1]

        return self

    def expand(self, excepted_len: int, expand_state: Optional[State] = None) -> None:
        cur_size = len(self.state_list)
        if cur_size >= excepted_len:
            return
        else:
            if expand_state is None:
                expand_state = self.state_list[-1]
            for _ in range(excepted_len - cur_size):
                self.state_list.append(expand_state)

    def to_list(self, is_vertical: bool = True) -> List:
        if is_vertical is True:
            states = [[],[],[],[]]
            for state in self.state_list:
                states[0].append(state.x)
                states[1].append(state.y)
                states[2].append(state.yaw)
                states[3].append(state.v)
        else:
            states = []
            for state in self.state_list:
                states.append([state.x, state.y, state.yaw, state.v])

        return states

    def __len__(self) -> int:
        return len(self.state_list)

    def __getitem__(self, key: int) -> State:
        return self.state_list[key]

    def __setitem__(self, key: int, value: State) -> None:
        self.state_list[key] = value


class Node:
    MAX_LEVEL: int = 6
    calc_value_callback = None

    def __init__(self, state = State(), level = 0, p: Optional["Node"] = None,
                 action: Optional[Action] = None, others: StateList = StateList(),
                 goal: State = State()) -> None:
        self.state: State = state

        self.value: float = 0
        self.reward: float = 0
        self.visits: int = 0
        self.action: Action = action
        self.parent: Node = p
        self.cur_level: int = level
        self.goal_pos: State = goal

        self.children: List[Node] = []
        self.actions: List[Action] = []
        self.other_agent_state: StateList = others

    @property
    def is_terminal(self) -> bool:
        return self.cur_level >= Node.MAX_LEVEL

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.children) >= len(Action)

    @staticmethod
    def initialize(max_level, callback) -> None:
        Node.MAX_LEVEL = max_level
        Node.calc_value_callback = callback

    def add_child(self, next_action: Action, delta_t: float, others: List[State] = []) -> "Node":
        new_state = kinematic_propagate(self.state, next_action.value, delta_t)
        node = Node(new_state, self.cur_level + 1, self, next_action, others, self.goal_pos)
        node.actions = self.actions + [next_action]
        Node.calc_value_callback(node, self.value)
        self.children.append(node)

        return node

    def next_node(self, delta_t: float, others: StateList = StateList()) -> "Node":
        next_action = random.choice(ActionList)
        new_state = kinematic_propagate(self.state, next_action.value, delta_t)
        node = Node(new_state, self.cur_level + 1, None, next_action, others, self.goal_pos)
        Node.calc_value_callback(node, self.value)

        return node

    def __repr__(self):
        return (f"children: {len(self.children)}, visits: {self.visits}, "
                f"reward: {self.reward}, actions: {self.actions}")


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


def kinematic_propagate(state: State, act: List[float], dt: float) -> State:
    next_state = State()
    acc, omega = act[0], act[1]

    next_state.x = state.x + state.v * np.cos(state.yaw) * dt
    next_state.y = state.y + state.v * np.sin(state.yaw) * dt
    next_state.v = state.v + acc * dt
    next_state.yaw = state.yaw + omega * dt

    while next_state.yaw > 2 * np.pi:
        next_state.yaw -= 2 * np.pi
    while next_state.yaw < 0:
        next_state.yaw += 2 * np.pi

    if next_state.v > 20:
        next_state.v = 20
    elif next_state.v < -20:
        next_state.v = -20

    return next_state
