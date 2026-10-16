# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# analyze_experiment2.py
import gc
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

# ---------------------------
# 绘图风格设置
# ---------------------------
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18

# ---------------------------
# 参数配置（与实验二计算时一致）
# ---------------------------
alphas = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
beta_Us = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

# 固定参数（与计算时保持一致）
N = 1000
m_consciousness = 4
m_risk = 2
T = 20.0
N_steps = 2048


# ---------------------------
# 数据加载函数
# ---------------------------
def load_experiment2_data(output_dir=_os.path.join(_DATA, "results_fractional")):
    """加载实验二的所有数据"""
    data_dict = {}

    for alpha in alphas:
        for beta_U in beta_Us:
            case_dir = os.path.join(output_dir, f"alpha_{alpha}_betaU_{beta_U}")

            # 检查数据是否存在
            if not os.path.exists(os.path.join(case_dir, "y.npy")):
                print(f"Warning: Data missing for alpha={alpha}, beta_U={beta_U}")
                continue

            # 加载数据
            t = np.load(os.path.join(case_dir, "t.npy"))
            y = np.load(os.path.join(case_dir, "y.npy"))

            # 计算群体平均感染密度
            I_avg = compute_infection_density(y, N)

            # 存储数据
            key = (alpha, beta_U)
            data_dict[key] = {
                't': t,
                'y': y,
                'I_avg': I_avg,
                'alpha': alpha,
                'beta_U': beta_U
            }

            print(f"Loaded: alpha={alpha}, beta_U={beta_U}")

    return data_dict


def compute_infection_density(y, N):
    """计算感染密度（I状态的群体平均值）"""
    t_steps = y.shape[0]
    I_avg = np.zeros(t_steps)

    for t in range(t_steps):
        state_matrix = y[t].reshape(N, 3)
        I_avg[t] = np.mean(state_matrix[:, 2])  # I状态在第3列

    return I_avg


# ---------------------------
# 分析函数
# ---------------------------
def compute_settling_time(I_avg, t, tolerance=0.05):
    """
    计算5%调节时间
    定义为第一次进入终值±5%范围内且不再超出所需的时间
    """
    final_value = I_avg[-1]
    tolerance_band = tolerance * final_value

    # 找到第一次进入容忍带的时间
    in_band = np.abs(I_avg - final_value) <= tolerance_band

    # 从后往前找到最后一个离开容忍带的时间点
    # 然后取之后第一次进入容忍带的时间
    settled_indices = np.where(in_band)[0]
    if len(settled_indices) == 0:
        return t[-1]  # 如果没有进入容忍带，返回总时间

    # 找到持续停留在容忍带的起始点
    for i in range(len(settled_indices) - 1, 0, -1):
        if settled_indices[i] - settled_indices[i - 1] > 1:
            first_settled_index = settled_indices[i]
            break
    else:
        first_settled_index = settled_indices[0]

    return t[first_settled_index]


# ---------------------------
# 绘图函数
# ---------------------------
def plot_betaU_trajectories(data_dict, output_dir):
    """为每个beta_U绘制不同alpha的感染密度轨迹"""
    b=2
    # 设置全局字体大小（影响所有文本）
    plt.rcParams.update({
        'font.size': 14*b,  # 全局字体大小
        'axes.titlesize': 16*b,  # 标题字体大小
        'axes.labelsize': 15*b,  # 坐标轴标签字体大小
        'xtick.labelsize': 13*b,  # x轴刻度标签字体大小
        'ytick.labelsize': 13*b,  # y轴刻度标签字体大小
        'legend.fontsize': 12*b,  # 图例字体大小
        'figure.titlesize': 18*b  # 图形标题字体大小
    })

    # 颜色和线型设置
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    # 计算所有数据中的最大和最小感染密度
    global_min = float('inf')
    global_max = float('-inf')
    for data in data_dict.values():
        global_min = min(global_min, np.min(data['I_avg']))
        global_max = max(global_max, np.max(data['I_avg']))

    # 计算所有数据中的最大跨度
    max_span = 0
    for beta_U in beta_Us:
        # 对于每个beta_U，计算该组数据的最小和最大值
        beta_min = float('inf')
        beta_max = float('-inf')
        for alpha in alphas:
            key = (alpha, beta_U)
            if key in data_dict:
                data = data_dict[key]
                beta_min = min(beta_min, np.min(data['I_avg']))
                beta_max = max(beta_max, np.max(data['I_avg']))

        # 更新最大跨度
        span = beta_max - beta_min
        if span > max_span:
            max_span = span

    # 为最大跨度添加一些边距
    max_span *= 1.1

    for beta_idx, beta_U in enumerate(beta_Us):
        # 增大图形尺寸但保持比例（宽度从6增加到8，高度从5增加到6.67）
        fig, ax = plt.subplots(figsize=(8, 6.67))  # 比例保持 6:5 = 1.2，这里 8:6.67 ≈ 1.2

        # 增加线条宽度，使其在放大后更清晰
        line_width = 2.5

        # 计算当前beta_U组数据的最小和最大值
        beta_min = float('inf')
        beta_max = float('-inf')
        for alpha in alphas:
            key = (alpha, beta_U)
            if key in data_dict:
                data = data_dict[key]
                beta_min = min(beta_min, np.min(data['I_avg']))
                beta_max = max(beta_max, np.max(data['I_avg']))

        # 计算居中显示的范围
        center = (beta_min + beta_max) / 2
        y_min = max(0, center - max_span / 2)  # 确保不低于0
        y_max = min(1, center + max_span / 2)  # 确保不超过1

        # 如果范围太小，调整以确保使用完整的跨度
        if (y_max - y_min) < max_span:
            if y_min == 0:
                y_max = min(1, max_span)
            elif y_max == 1:
                y_min = max(0, 1 - max_span)
            else:
                # 居中调整
                y_min = center - max_span / 2
                y_max = center + max_span / 2

        # 绘制曲线
        for alpha_idx, alpha in enumerate(alphas):
            key = (alpha, beta_U)
            if key in data_dict:
                data = data_dict[key]
                t = data['t']
                I_avg = data['I_avg']

                # 跳过初始的瞬态阶段以获得更好的可视化
                mask = t > 0.01

                ax.semilogx(t[mask], I_avg[mask],
                            color=colors[alpha_idx],
                            linestyle=linestyles[alpha_idx % len(linestyles)],
                            linewidth=line_width,
                            label=f'α={alpha}')

        # 设置坐标轴标签，并确保字体大小
        ax.set_xlabel('Time', fontsize=32)  # 单独设置确保大小
        ax.set_ylabel('$\\rho^{AI}$', fontsize=32)

        # 设置刻度标签大小
        ax.tick_params(axis='both', which='major', labelsize=28)

        # 如果需要标题，可以取消注释并调整大小
        # ax.set_title(f'Infection Dynamics for βU = {beta_U}', fontsize=18)

        ax.grid(True, alpha=0.3)

        # 如果需要图例，可以取消注释并调整大小和位置
        # ax.legend(bbox_to_anchor=(0.95, 1), loc='upper left', fontsize=24)
        # 或者
        # ax.legend(loc='best', fontsize=24, framealpha=0.9)
        # 论文 Fig. 5(b) 只在第一个面板 (betaU = 0.3) 显示图例，其余面板不显示；
        # 这里按论文排版恢复该图例，使重跑结果与论文图完全一致。
        if beta_U == 0.3:
            ax.legend(loc='best', fontsize=24, framealpha=0.9)

        # 设置统一的y轴跨度，但根据数据居中
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'betaU_{beta_U}_trajectories.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'betaU_{beta_U}_trajectories.pdf'),
                    bbox_inches='tight')
        plt.show()
        plt.close()

    # 重置matplotlib默认设置，避免影响其他图形
    plt.rcParams.update(plt.rcParamsDefault)
def plot_settling_time_heatmap(settling_times, output_dir):
    """绘制调节时间热力图"""

    # 创建矩阵用于热力图
    settling_matrix = np.zeros((len(alphas), len(beta_Us)))

    for i, alpha in enumerate(alphas):
        for j, beta_U in enumerate(beta_Us):
            settling_matrix[i, j] = settling_times.get((alpha, beta_U), np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用seaborn绘制热力图，使用对数颜色标度
    im = sns.heatmap(settling_matrix,
                     xticklabels=beta_Us,
                     yticklabels=alphas,
                     annot=True, fmt='.2f',
                     cmap='viridis_r',  # 反向颜色映射，红色表示长时间
                     cbar_kws={'label': 'Settling Time (5% tolerance)'},
                     ax=ax)

    ax.set_xlabel('βU')
    ax.set_ylabel('α')
    ax.set_title('5% Settling Time for Different α and βU')

    # 改善标注外观
    for text in im.texts:
        text.set_size(10)
        text.set_weight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'settling_time_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'settling_time_heatmap.pdf'),
                bbox_inches='tight')
    plt.show()
    plt.close()


# ---------------------------
# 主分析函数
# ---------------------------
def analyze_experiment2(output_dir=_os.path.join(_DATA, "results_fractional"), analysis_dir=_os.path.join(_DATA, "analysis_experiment2")):
    """主分析函数"""

    os.makedirs(analysis_dir, exist_ok=True)

    print("=== Loading Experiment 2 Data ===")
    data_dict = load_experiment2_data(output_dir)

    if not data_dict:
        print("No data found! Please check the output directory.")
        return

    print(f"\n=== Loaded {len(data_dict)} parameter combinations ===")

    # 计算调节时间
    print("\n=== Computing Settling Times ===")
    settling_times = {}

    for key, data in data_dict.items():
        settling_time = compute_settling_time(data['I_avg'], data['t'])
        settling_times[key] = settling_time
        print(f"α={data['alpha']}, βU={data['beta_U']}: settling time = {settling_time:.4f}")

    # 绘制轨迹图
    print("\n=== Plotting Beta_U Trajectories ===")
    plot_betaU_trajectories(data_dict, analysis_dir)

    # 绘制调节时间热力图
    print("\n=== Plotting Settling Time Heatmap ===")
    plot_settling_time_heatmap(settling_times, analysis_dir)

    # 保存分析结果
    print("\n=== Saving Analysis Results ===")
    summary_data = {}
    for key, data in data_dict.items():
        alpha, beta_U = key
        summary_key = f"alpha_{alpha}_betaU_{beta_U}"
        summary_data[summary_key] = {
            'final_infection': float(data['I_avg'][-1]),
            'settling_time': float(settling_times[key]),
            'alpha': alpha,
            'beta_U': beta_U
        }

    with open(os.path.join(analysis_dir, "experiment2_analysis_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n=== Experiment 2 Analysis Completed ===")
    print(f"Results saved to: {analysis_dir}")

    return data_dict, settling_times


if __name__ == "__main__":
    analyze_experiment2()
