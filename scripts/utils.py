import numpy as np


class Node:
     def __init__(self, x = 0, y = 0, yaw = 0, v = 0, action = None, p = None):
         self.x = x
         self.y = y
         self.yaw = yaw
         self.v = v
         self.action = action
         self.reward = 0
         self.parent = p


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


def kinematic_propagate(node:Node, act, dt):
        acc = act[0]
        omega = act[1]

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
