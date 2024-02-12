import numpy as np
import matplotlib.pyplot as plt


class Vehicle:
    def __init__(self, x, y, yaw, length = 5, width = 2, color = 'k'):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = 0
        self.length = length
        self.width = width
        self.color = color
        self.vehicle_box2d = self.get_box2d()


    def get_box2d(self):
        vehicle = np.array(
            [[-self.length/2, self.length/2, self.length/2, -self.length/2, -self.length/2],
            [self.width/2, self.width/2, -self.width/2, -self.width/2, self.width/2]]
        )
        rot1 = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                     [np.sin(self.yaw), np.cos(self.yaw)]])

        vehicle = np.dot(rot1, vehicle)
        vehicle += np.array([[self.x], [self.y]])

        return vehicle


    def update(self, acc, omega, dt):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.v += acc * dt
        self.yaw += omega * dt

        self.vehicle_box2d = self.get_box2d()


    def draw_vehicle(self):
        head = np.array(
            [[0.3 * self.length, 0.3 * self.length], [self.width/2, -self.width/2]]
        )
        rot1 = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                     [np.sin(self.yaw), np.cos(self.yaw)]])
        head = np.dot(rot1, head)
        head += np.array([[self.x], [self.y]])

        self.vehicle_box2d = self.get_box2d()

        plt.plot(self.vehicle_box2d[0, :], self.vehicle_box2d[1, :], self.color)
        plt.plot(head[0, :], head[1, :], self.color)


    def has_overlap(self, box2d) -> bool:
        box2d_sides_num = len(box2d[0])
        total_sides = []
        for i in range(1, len(self.vehicle_box2d[0])):
            vec_x = self.vehicle_box2d[0][i] - self.vehicle_box2d[0][i - 1]
            vec_y = self.vehicle_box2d[1][i] - self.vehicle_box2d[1][i - 1]
            total_sides.append([vec_x, vec_y])
        for i in range(1, box2d_sides_num):
            vec_x = box2d[0][i] - box2d[0][i - 1]
            vec_y = box2d[1][i] - box2d[1][i - 1]
            total_sides.append([vec_x, vec_y])

        for i in range(len(total_sides)):
            separating_axis = [total_sides[i][0], -total_sides[i][1]]

            vehicle_min = np.inf
            vehicle_max = -np.inf
            for j in range(1, len(self.vehicle_box2d[0])):
                project = separating_axis[0] * self.vehicle_box2d[0][j] + separating_axis[1] * self.vehicle_box2d[1][j]
                vehicle_min = min(vehicle_min, project)
                vehicle_max = max(vehicle_max, project)
            
            box2d_min = np.inf
            box2d_max = -np.inf
            for j in range(0, box2d_sides_num):
                project = separating_axis[0] * box2d[0][j] + separating_axis[1] * box2d[1][j]
                box2d_min = min(box2d_min, project)
                box2d_max = max(box2d_max, project)

            if vehicle_min > box2d_max or box2d_min > vehicle_max:
                return False
    
        return True
    

if __name__ == "__main__":
    print("has_overlap() test ...")
    print("========================================")
    vehicle = Vehicle(-5, -5, np.pi / 2)
    rect = [
            [[-25, -25, -2*4, -4, -4, -25],
                [-25, -4, -4, -2*4, -25, -25]],
            [[25, 25, 2*4, 4, 4, 25],
                [-25, -4, -4, -2*4, -25, -25]],
        ]
    laneline = [
        [[0, 0], [25, 0]],
        [[0, -15], [-7, -7]],
        [[5, -25], [5, -25]]
    ]

    bool_has_overlap = vehicle.has_overlap(rect[0])
    print("has_overlap with rect[0]: ", bool_has_overlap)
    bool_has_overlap = vehicle.has_overlap(rect[1])
    print("has_overlap with rect[1]: ", bool_has_overlap)
    bool_has_overlap = vehicle.has_overlap(laneline[0])
    print("has_overlap with laneline[0]: ", bool_has_overlap)
    bool_has_overlap = vehicle.has_overlap(laneline[1])
    print("has_overlap with laneline[1]: ", bool_has_overlap)
    bool_has_overlap = vehicle.has_overlap(laneline[2])
    print("has_overlap with laneline[2]: ", bool_has_overlap)

    plt.fill(*rect[0], color='gray', alpha=0.5)
    plt.fill(*rect[1], color='gray', alpha=0.5)
    plt.plot(*laneline[0], linestyle='--', color='orange', linewidth=2)
    plt.plot(*laneline[1], linestyle='--', color='orange', linewidth=2)
    plt.plot(*laneline[2], linestyle='--', color='orange', linewidth=2)
    vehicle.draw_vehicle()

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.gca().set_aspect('equal')
    plt.show()
