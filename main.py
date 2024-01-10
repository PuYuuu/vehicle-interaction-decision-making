import matplotlib.pyplot as plt

rect0 = [[-25, -25, -8, -4, -4, -25], [-25, -4, -4, -8, -25, -25]]
rect1 = [[25, 25, 8, 4, 4, 25], [-25, -4, -4, -8, -25, -25]]
rect2 = [[25, 25, 8, 4, 4, 25], [25, 4, 4, 8, 25, 25]]
rect3 = [[-25, -25, -8, -4, -4, -25], [25, 4, 4, 8, 25, 25]]

laneline0 = [[0, 0], [-25, -8]]
laneline1 = [[0, 0], [8, 25]]
laneline2 = [[-25, -8], [0, 0]]
laneline3 = [[8, 25], [0, 0]]

# 绘制矩形的填充区域
plt.fill(*rect0, color='gray', alpha=0.5)
plt.fill(*rect1, color='gray', alpha=0.5)
plt.fill(*rect2, color='gray', alpha=0.5)
plt.fill(*rect3, color='gray', alpha=0.5)
plt.plot(*rect0, color='k', linewidth=2)
plt.plot(*rect1, color='k', linewidth=2)
plt.plot(*rect2, color='k', linewidth=2)
plt.plot(*rect3, color='k', linewidth=2)

plt.plot(*laneline0, linestyle='--', color='orange', linewidth=2)
plt.plot(*laneline1, linestyle='--', color='orange', linewidth=2)
plt.plot(*laneline2, linestyle='--', color='orange', linewidth=2)
plt.plot(*laneline3, linestyle='--', color='orange', linewidth=2)

plt.xlim(-25, 25)
plt.ylim(-25, 25)
plt.gca().set_aspect('equal')
# plt.axis("equal")
# 显示图形
plt.show()
