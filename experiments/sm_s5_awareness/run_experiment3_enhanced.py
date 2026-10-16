# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_experiment3_enhanced.py
"""
实验3增强版：使用3D多层柱状图可视化意识层影响
"""

import os
import json
import numpy as np
from run_experiment3 import run_experiment3, load_existing_results, lambd_values, delta_values, m_consciousness_values
from multi_layer_3d_plot import MultiLayer3DPlot


def create_3d_visualization(results_matrices, output_dir):
    """创建3D多层柱状图可视化"""

    # 准备层标签
    layer_labels = [f'm={m}' for m in m_consciousness_values]

    # 准备数据矩阵列表
    data_matrices = [results_matrices[m] for m in m_consciousness_values]

    # 创建3D可视化 - 使用瘦高的图形比例
    plotter = MultiLayer3DPlot(lambd_values, delta_values, layer_labels)

    # 使用更大的高度和z轴缩放，创建瘦高效果
    fig, ax = plotter.create_plot(figsize=(16, 20), z_scale=1.5)

    # 绘制数据 - 增加层间距和z轴数据缩放
    plotter.plot_data(data_matrices,
                      bar_width=0.9,
                      bar_depth=0.9,
                      layer_spacing=10,  # 增加层间距
                      alpha=0.8,
                      edge_alpha=0.5,
                      z_scale=5)  # 增加z轴数据缩放

    # 设置标题
    # ax.set_title('Impact of Consciousness Layer on Infection Dynamics\n'
    #              '3D Multi-Layer Visualization of Final Infection Densities',
    #              fontsize=16, fontweight='bold', pad=40)

    # 添加说明文本
    ax.text2D(-0.1, 0.98,
              'Parameters:\n• $\\alpha$ = 0.8\n• $\\beta^U$ = 0.4\n• $\gamma$ = 1/3\n• $\mu$ = 1.0',
              transform=ax.transAxes, fontsize=22, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存图像
    plotter.save_plot(os.path.join(output_dir, 'experiment3_3d_multilayer_tall.png'), dpi=300)
    plotter.save_plot(os.path.join(output_dir, 'experiment3_3d_multilayer_tall.pdf'))

    # 从不同角度保存多个视图
    angles = [
        (25, 45),  # 标准视角
        (15, 60),  # 稍微降低仰角，增加旋转
        (35, 30),  # 提高仰角，减少旋转
        (20, 120),  # 侧面视角
    ]

    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev=elev, azim=azim)
        plotter.save_plot(os.path.join(output_dir, f'experiment3_3d_view_{i + 1}.png'), dpi=300)

    plotter.show()

    return plotter


def analyze_and_visualize_experiment3():
    """分析并可视化实验3结果"""

    output_dir = _os.path.join(_DATA, "results_experiment3")

    # 加载已有结果
    results_matrices = load_existing_results(output_dir)

    if len(results_matrices) != len(m_consciousness_values):
        print("部分结果缺失，请先运行完整的实验3")
        return

    # 创建3D可视化
    print("创建3D多层柱状图（瘦高版）...")
    plotter = create_3d_visualization(results_matrices, output_dir)

    # 分析结果
    print("\n分析实验结果...")
    from run_experiment3 import analyze_results
    analysis_results = analyze_results(results_matrices, output_dir)

    # 创建简化的2D热力图作为补充
    create_simplified_heatmaps(results_matrices, output_dir)

    print(f"\n=== 实验3增强可视化完成 ===")
    print(f"结果保存在: {output_dir}")


def create_simplified_heatmaps(results_matrices, output_dir):
    """创建简化的2D热力图作为补充"""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(m_consciousness_values), figsize=(15, 5))
    if len(m_consciousness_values) == 1:
        axes = [axes]

    # 全局颜色范围
    all_data = np.concatenate([matrix.flatten() for matrix in results_matrices.values()])
    vmin, vmax = np.min(all_data), np.max(all_data)

    for idx, m in enumerate(m_consciousness_values):
        ax = axes[idx]
        matrix = results_matrices[m]

        im = ax.imshow(matrix, cmap='viridis', aspect='auto',
                       extent=[delta_values[0], delta_values[-1],
                               lambd_values[-1], lambd_values[0]],
                       vmin=vmin, vmax=vmax)

        ax.set_xlabel('δ', fontsize=24)
        if idx == 0:
            ax.set_ylabel('λ', fontsize=24)
        ax.set_title(f'm = {m}', fontsize=28, fontweight='bold')

        # 添加数值标签（只显示极值）
        for i in range(len(lambd_values)):
            for j in range(len(delta_values)):
                value = matrix[i, j]
                if value >= 0:  # 有效数据
                    # 只在接近极值时显示数值
                    if abs(value - vmin) < 0.05 or abs(value - vmax) < 0.05:
                        color = 'white' if value > (vmin + vmax) / 2 else 'black'
                        ax.text(delta_values[j], lambd_values[i], f'{value:.2f}',
                                ha='center', va='center', color=color, fontsize=16)

    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label('Final Infection Density', fontsize=24)

    plt.savefig(os.path.join(output_dir, 'simplified_heatmaps.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'simplified_heatmaps.pdf'),
                bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    analyze_and_visualize_experiment3()