import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# 全局参数设置
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'figure.dpi': 150,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'savefig.facecolor': 'white'
})

# =============== 生成三维正态分布点云 ===============
np.random.seed(42)
n = 1000  # 点数量
mu = np.array([1, 2, -1])  # 三维均值
SIGMA = np.array([[3, 1, 1.5],
                  [1, 2, 0.5],
                  [1.5, 0.5, 2]])  # 三维协方差

data = np.random.multivariate_normal(mu, SIGMA, size=n)  # shape: (n, 3)
x, y, z = data[:, 0], data[:, 1], data[:, 2]

# =============== 绘制3D点云 ===============
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z,
                c=z,           # 颜色随z变化
                cmap='viridis',
                s=20, alpha=0.8,
                edgecolors='w', linewidth=0.2)

ax.set_xlabel('X', fontweight='bold', labelpad=10, fontsize=14)
ax.set_ylabel('Y', fontweight='bold', labelpad=10, fontsize=14)
ax.set_zlabel('Z', fontweight='bold', labelpad=10, fontsize=14)
ax.set_title('3D Multivariate Normal Distribution', pad=15, fontsize=18)
ax.grid(True, linestyle=':', alpha=0.3)
ax.set_xlim(mu[0] - 4*np.sqrt(SIGMA[0,0]), mu[0] + 4*np.sqrt(SIGMA[0,0]))
ax.set_ylim(mu[1] - 4*np.sqrt(SIGMA[1,1]), mu[1] + 4*np.sqrt(SIGMA[1,1]))
ax.set_zlim(mu[2] - 4*np.sqrt(SIGMA[2,2]), mu[2] + 4*np.sqrt(SIGMA[2,2]))

ax.view_init(elev=25, azim=30)
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label='Z Value')

# =============== 3D旋转动画 ===============
def rotate(angle):
    ax.view_init(elev=25, azim=angle)
    return fig,

ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50, blit=False)

# =============== 保存GIF ===============
gif_path = '3d_normal.gif'
print("正在生成3D点云GIF动画...")
writer = PillowWriter(fps=15, metadata=dict(artist='3D Normal'), bitrate=1800)
ani.save(gif_path, writer=writer, dpi=100)
print(f"GIF已保存至: {gif_path}\n可插入PPT或网页动态演示")

plt.close(fig)
