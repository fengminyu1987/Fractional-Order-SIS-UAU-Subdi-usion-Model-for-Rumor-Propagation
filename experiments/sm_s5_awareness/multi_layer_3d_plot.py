# multi_layer_3d_plot.py
"""
三维多层柱状图可视化工具
用于展示多个参数平面上的数据分布
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib import cm


class MultiLayer3DPlot:
    """三维多层柱状图类"""

    def __init__(self, x_labels, y_labels, layer_labels):
        """
        初始化

        Parameters:
        -----------
        x_labels : list
            x轴标签（参数值）
        y_labels : list
            y轴标签（参数值）
        layer_labels : list
            各层的标签
        """
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.layer_labels = layer_labels
        self.n_layers = len(layer_labels)
        self.n_x = len(x_labels)
        self.n_y = len(y_labels)

        # 颜色映射
        self.cmap = cm.viridis  # 使用viridis颜色映射，对色盲友好

        # 图形设置
        self.fig = None
        self.ax = None

    def create_plot(self, figsize=(12, 14), z_scale=2.0):
        """
        创建3D图形

        Parameters:
        -----------
        figsize : tuple
            图形尺寸 (宽度, 高度)
        z_scale : float
            z轴缩放因子，用于创建瘦高的效果
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 设置z轴缩放，创建瘦高效果
        self.ax.set_box_aspect([1, 1, z_scale])  # z轴是x和y轴的z_scale倍

        return self.fig, self.ax

    def plot_data(self, data_matrices, bar_width=0.8, bar_depth=0.8,
                  layer_spacing=1.5, alpha=0.8, edge_color='black',
                  edge_alpha=0.3, edge_width=0.5, z_scale=1.0):
        """
        绘制多层3D柱状图

        Parameters:
        -----------
        data_matrices : list of 2D arrays
            每个层的数据矩阵，形状为 (n_x, n_y)
        bar_width : float
            柱子的宽度（x方向）
        bar_depth : float
            柱子的深度（y方向）
        layer_spacing : float
            层之间的间距
        alpha : float
            柱子透明度
        edge_color : str
            柱子边缘颜色
        edge_alpha : float
            柱子边缘透明度
        edge_width : float
            柱子边缘宽度
        z_scale : float
            z轴数据缩放因子，用于调整柱子的相对高度
        """
        if len(data_matrices) != self.n_layers:
            raise ValueError(f"数据矩阵数量({len(data_matrices)})与层数({self.n_layers})不匹配")

        # 计算全局数据范围用于颜色映射
        all_data = np.concatenate([matrix.flatten() for matrix in data_matrices])
        vmin, vmax = np.min(all_data), np.max(all_data)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # 为每层创建网格
        x_pos = np.arange(self.n_x)
        y_pos = np.arange(self.n_y)
        x_mesh, y_mesh = np.meshgrid(x_pos, y_pos, indexing='ij')

        # 绘制每一层
        for layer_idx in range(self.n_layers):
            z_offset = layer_idx * layer_spacing
            data_matrix = data_matrices[layer_idx]

            # 绘制该层的所有柱子
            for i in range(self.n_x):
                for j in range(self.n_y):
                    value = data_matrix[i, j]
                    if np.isnan(value) or value < 0:  # 跳过无效数据
                        continue

                    # 柱子位置（稍微缩小以避免重叠）
                    x = x_mesh[i, j] - (bar_width * 0.9) / 2
                    y = y_mesh[i, j] - (bar_depth * 0.9) / 2
                    z = z_offset

                    # 柱子尺寸 - 应用z_scale调整高度
                    dx = bar_width * 0.9
                    dy = bar_depth * 0.9
                    dz = value * z_scale  # 应用z轴缩放

                    # 柱子颜色
                    color = self.cmap(norm(value))

                    # 绘制柱子
                    self.ax.bar3d(x, y, z, dx, dy, dz,
                                  color=color, alpha=alpha,
                                  edgecolor=edge_color, linewidth=edge_width,
                                  zsort='average')

        # 设置坐标轴标签
        self.ax.set_xlabel('$\lambda$', fontsize=24, labelpad=10)
        self.ax.set_ylabel('$\delta$', fontsize=24, labelpad=10)
        self.ax.set_zlabel('$\\rho^{AI**}$', fontsize=24, labelpad=15)

        # 设置坐标轴刻度
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels([f'{x:.1f}' for x in self.x_labels])
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels([f'{y:.1f}' for y in self.y_labels])

        # 设置z轴刻度为层标签
        z_ticks = np.arange(self.n_layers) * layer_spacing
        self.ax.set_zticks(z_ticks)
        self.ax.set_zticklabels(self.layer_labels)

        # 设置刻度标签字体大小（假设原来默认是10，现在放大两倍为20）
        self.ax.tick_params(axis='x', labelsize=15)
        self.ax.tick_params(axis='y', labelsize=15)
        self.ax.tick_params(axis='z', labelsize=13)

        # 设置z轴范围，确保所有层都可见
        z_max = (self.n_layers - 1) * layer_spacing + vmax * z_scale
        self.ax.set_zlim(0, z_max)

        # 添加颜色条
        sm = cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.6, aspect=20, pad=0.1)
        # 设置颜色条刻度字体大小
        cbar.ax.tick_params(labelsize=14)  # 添加这行，设置刻度标签字体大小
        cbar.set_label('$\\rho^{AI**}$', rotation=270, labelpad=15, fontsize=24)

        # 设置视角以获得最佳可视化效果
        self.ax.view_init(elev=25, azim=45)

        # 添加网格
        self.ax.grid(True, alpha=0.3)

        return self.fig, self.ax

    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        """保存图像"""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            print(f"图像已保存至: {filename}")
        else:
            print("错误: 未创建图形")

    def show(self):
        """显示图像"""
        if self.fig is not None:
            plt.show()
        else:
            print("错误: 未创建图形")


# 测试函数
def test_multi_layer_plot():
    """测试多层3D柱状图"""

    # 测试数据
    x_labels =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # λ值
    y_labels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # δ值
    layer_labels = ['m=2', 'm=4', 'm=8']  # 层标签

    # 生成测试数据矩阵
    np.random.seed(42)
    data_matrices = []

    for layer_idx, layer_label in enumerate(layer_labels):
        # 为每层生成不同的数据模式
        base_matrix = np.zeros((len(x_labels), len(y_labels)))

        for i, x in enumerate(x_labels):
            for j, y in enumerate(y_labels):
                # 模拟感染密度：高λ和低δ导致低感染，低λ和高δ导致高感染
                base_value = 0.8 - 0.6 * x + 0.4 * y
                # 添加层相关的变化
                layer_effect = 0.1 * (layer_idx + 1)
                # 添加随机噪声
                noise = np.random.normal(0, 0.05)

                value = max(0, min(1, base_value + layer_effect + noise))
                base_matrix[i, j] = value

        data_matrices.append(base_matrix)

    # 创建可视化 - 使用瘦高的图形比例
    plotter = MultiLayer3DPlot(x_labels, y_labels, layer_labels)
    plotter.create_plot(figsize=(14, 16), z_scale=1.5)  # 增加高度和z轴缩放
    plotter.plot_data(data_matrices, bar_width=0.9, bar_depth=0.9,
                      layer_spacing=10, alpha=0.8, z_scale=1.5)  # 增加层间距和z轴数据缩放

    plotter.ax.set_title('3D Multi-Layer Bar Chart Test\nEffect of Awareness Layer on Infection Dynamics',
                         fontsize=16, fontweight='bold', pad=30)

    # 保存测试图像
    plotter.save_plot('test_multi_layer_3d_tall.png', dpi=300)
    plotter.show()

    return data_matrices


if __name__ == "__main__":
    print("测试多层3D柱状图（瘦高版）...")
    test_data = test_multi_layer_plot()
    print("测试完成！")