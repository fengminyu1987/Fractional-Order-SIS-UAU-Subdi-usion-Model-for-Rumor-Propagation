# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_experiment3.py
"""
实验3：意识层对传播动态的影响
研究不同意识层网络结构和参数对最终感染密度的影响
"""

import gc
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm

from fast_caputo_solver import solve_caputo_blocked, BlockConfig

# ---------------------------
# 参数配置
# ---------------------------
# 实验3的固定参数
alpha = 0.8
N = 250
m_risk = 2
T = 20.0
N_steps = 2048 *2

# 风险层固定参数（使用保守参数）
risk_params = {
    'beta_U': 1,  # 保守的感染率
    'gamma': 1 / 3,  # 意识者的感染率降低因子
    'mu': 1.0  # 恢复率
}

# 意识层参数扫描范围
lambd_values = np.round(np.linspace(0.1, 1.0, 10), 2)  # 意识传播率
delta_values = np.round(np.linspace(0.1, 1.0, 10), 2)  # 遗忘率

# 意识层网络结构（不同的平均度）
m_consciousness_values = [2, 4, 8]


# ---------------------------
# 初值生成
# ---------------------------
def generate_initial_conditions(N, infected_fraction=0.02, aware_fraction=0.03):
    """生成初始条件"""
    y0 = np.zeros(3 * N)

    for i in range(N):
        # 初始大部分为 US
        p_US = 1.0 - infected_fraction - aware_fraction
        p_AS = aware_fraction
        p_AI = infected_fraction

        # 添加小随机扰动
        perturbation = np.random.normal(0, 0.01, 3)
        p_vec = np.array([p_US, p_AS, p_AI]) + perturbation
        p_vec = np.clip(p_vec, 0, 1)  # 确保概率在 [0,1] 范围内
        p_vec = p_vec / np.sum(p_vec)  # 重新归一化

        y0[3 * i:3 * i + 3] = p_vec

    return y0


# ---------------------------
# 方程右端
# ---------------------------
def make_F(A, B, params):
    """创建方程右端函数"""
    N = params["N"]
    beta_U, beta_A = params["beta_U"], params["beta_U"] * params["gamma"]
    lambd, delta, mu = params["lambd"], params["delta"], params["mu"]

    def F(P, t):
        M = len(P) - 1
        P_data = P[:-1, :].reshape(M, N, 3)
        BP, AP = np.zeros_like(P_data), np.zeros_like(P_data)
        for j in range(M):
            BP[j] = A @ P_data[j]
            AP[j] = B @ P_data[j]
        M3 = np.array([[0, 0, 0],
                       [delta, -delta, 0],
                       [0, mu, -mu]])
        Q = np.zeros((N, M, 3, 3))
        for i in range(N):
            e = np.zeros(N);
            e[i] = 1
            for j in range(M):
                s1 = e @ BP[j] @ np.array([[0], [1], [1]])
                s2 = e @ AP[j] @ np.array([[0], [0], [1]])
                M1 = s1 * np.array([[-lambd, lambd, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
                M2 = s2 * np.array([[-beta_U, 0, beta_U],
                                    [0, -beta_A, beta_A],
                                    [0, 0, 0]])
                Q[i, j] = M3 + M1 + M2
        p = P_data.transpose(1, 0, 2)
        result = np.zeros((M, 3 * N))
        for j in range(M):
            row_parts = [p[i, j] @ Q[i, j] for i in range(N)]
            result[j] = np.concatenate(row_parts)
        return result

    return F


# ---------------------------
# 计算最终感染密度
# ---------------------------
def compute_final_infection_density(y, N):
    """计算最终感染密度"""
    # 获取最终状态
    final_state = y[-1].reshape(N, 3)
    # AI状态的密度
    final_AI_density = np.mean(final_state[:, 2])
    return final_AI_density


# ---------------------------
# 运行单个参数组合
# ---------------------------
def run_single_simulation(m_consciousness, lambd, delta, output_dir):
    """运行单个参数组合的模拟"""

    # 创建唯一标识符
    case_id = f"m{m_consciousness}_lambd{lambd}_delta{delta}"
    result_file = os.path.join(output_dir, f"{case_id}.json")

    # 检查是否已存在结果
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            result = json.load(f)
        return result['final_AI_density']

    # 生成网络
    np.random.seed(42)  # 固定种子以确保网络一致性
    G_consciousness = nx.barabasi_albert_graph(N, m_consciousness, seed=42)
    G_risk = nx.barabasi_albert_graph(N, m_risk, seed=43)
    A = nx.to_numpy_array(G_consciousness)
    B = nx.to_numpy_array(G_risk)

    # 生成初值
    y0 = generate_initial_conditions(N)

    # 准备参数
    params = {
        'N': N,
        'lambd': lambd,
        'delta': delta,
        'beta_U': risk_params['beta_U'],
        'gamma': risk_params['gamma'],
        'mu': risk_params['mu'],
        'alpha': alpha,
        'm_consciousness': m_consciousness,
        'm_risk': m_risk,
        'T': T,
        'N_steps': N_steps
    }

    # 创建方程右端函数
    F = make_F(A, B, params)

    # 配置求解器
    cfg = BlockConfig(
        block_steps=16,
        min_block_steps=4,
        max_block_steps=4096,
        tol=1e-4,
        verbose=False,
        picard_omega=0.7,
        omega_backtrack=0.8
    )

    # 求解分数阶方程
    try:
        t0 = time.time()
        t, y = solve_caputo_blocked(F, alpha, y0, T, N_steps, cfg)
        solve_time = time.time() - t0

        # 计算最终感染密度
        final_AI_density = compute_final_infection_density(y, N)

        # 保存结果（只保存关键数据）
        result = {
            'final_AI_density': float(final_AI_density),
            'solve_time': solve_time,
            'params': params
        }

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        # 清理内存
        del F, y
        gc.collect()

        return final_AI_density

    except Exception as e:
        print(f"Error in {case_id}: {e}")
        # 保存错误信息
        result = {
            'final_AI_density': -1.0,  # 使用-1表示错误
            'error': str(e),
            'params': params
        }
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        return -1.0


# ---------------------------
# 运行实验3
# ---------------------------
def run_experiment3(output_dir=_os.path.join(_DATA, "results_experiment3")):
    """运行实验3"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存实验配置
    config = {
        'alpha': alpha,
        'N': N,
        'm_risk': m_risk,
        'T': T,
        'N_steps': N_steps,
        'risk_params': risk_params,
        'lambd_values': lambd_values.tolist(),
        'delta_values': delta_values.tolist(),
        'm_consciousness_values': m_consciousness_values
    }

    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print("=== 实验3：意识层对传播动态的影响 ===")
    print(f"参数扫描范围:")
    print(f"  λ: {lambd_values}")
    print(f"  δ: {delta_values}")
    print(f"  m_consciousness: {m_consciousness_values}")
    print(f"风险层固定参数: {risk_params}")

    # 为每个m_consciousness创建结果矩阵
    results_matrices = {}

    for m_consciousness in m_consciousness_values:
        print(f"\n--- 运行 m_consciousness = {m_consciousness} ---")

        # 创建子目录
        m_dir = os.path.join(output_dir, f"m{m_consciousness}")
        os.makedirs(m_dir, exist_ok=True)

        # 初始化结果矩阵
        result_matrix = np.zeros((len(lambd_values), len(delta_values)))

        # 进度条
        total_combinations = len(lambd_values) * len(delta_values)
        pbar = tqdm(total=total_combinations, desc=f"m={m_consciousness}")

        # 扫描参数空间
        for i, lambd in enumerate(lambd_values):
            for j, delta in enumerate(delta_values):
                # 运行模拟
                final_AI_density = run_single_simulation(m_consciousness, lambd, delta, m_dir)
                result_matrix[i, j] = final_AI_density

                pbar.update(1)

        pbar.close()

        # 保存结果矩阵
        results_matrices[m_consciousness] = result_matrix
        np.save(os.path.join(m_dir, "result_matrix.npy"), result_matrix)

        print(f"m={m_consciousness} 完成，平均感染密度: {np.mean(result_matrix[result_matrix >= 0]):.4f}")

    return results_matrices


# ---------------------------
# 绘制热力图
# ---------------------------
def plot_heatmaps(results_matrices, output_dir):
    """绘制热力图"""

    # 设置matplotlib样式
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11
    })

    # 创建子图
    fig, axes = plt.subplots(1, len(m_consciousness_values), figsize=(15, 5))

    # 如果只有一个子图，确保axes是列表
    if len(m_consciousness_values) == 1:
        axes = [axes]

    # 设置颜色映射 - 使用从绿色到红色的渐变，突出感染密度的变化
    cmap = plt.cm.RdYlGn_r  # 从绿色(低感染)到红色(高感染)

    # 找到所有有效数据的全局最小值和最大值，用于统一颜色条
    all_valid_data = []
    for matrix in results_matrices.values():
        valid_data = matrix[matrix >= 0]  # 排除错误值(-1)
        all_valid_data.extend(valid_data)

    vmin = min(all_valid_data)
    vmax = max(all_valid_data)

    for idx, m_consciousness in enumerate(m_consciousness_values):
        ax = axes[idx]
        matrix = results_matrices[m_consciousness]

        # 创建热力图
        im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                       extent=[delta_values[0], delta_values[-1],
                               lambd_values[-1], lambd_values[0]],
                       vmin=vmin, vmax=vmax)

        # 设置坐标轴
        ax.set_xlabel('Forgetting Rate (δ)', fontsize=24)
        if idx == 0:
            ax.set_ylabel('Awareness Rate (λ)', fontsize=24)

        ax.set_title(f'm = {m_consciousness}', fontsize=26, fontweight='bold')

        # 添加网格线
        ax.set_xticks(delta_values)
        ax.set_yticks(lambd_values)
        ax.grid(False)

        # # 在每个单元格中心添加数值标签
        # for i in range(len(lambd_values)):
        #     for j in range(len(delta_values)):
        #         value = matrix[i, j]
        #         if value >= 0:  # 只显示有效值
        #             color = 'white' if value > (vmin + vmax) / 2 else 'black'
        #             ax.text(delta_values[j], lambd_values[i], f'{value:.2f}',
        #                     ha='center', va='center', color=color, fontsize=8)

    # 添加颜色条
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label('Final Infection Density (AI)', fontsize=24)

    # 保存图像
    plt.savefig(os.path.join(output_dir, 'experiment3_heatmaps.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'experiment3_heatmaps.pdf'),
                bbox_inches='tight')
    plt.show()

    # 额外绘制一个汇总图，显示不同m值下的平均感染密度
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    # 创建均匀的x轴位置
    x_positions = np.arange(len(m_consciousness_values))

    y_max = 0
    for idx, m_consciousness in enumerate(m_consciousness_values):
        matrix = results_matrices[m_consciousness]
        valid_data = matrix[matrix >= 0]

        # 计算每个m值的平均感染密度
        avg_infection = np.mean(valid_data)
        std_infection = np.std(valid_data)
        y_max = max(y_max, avg_infection + std_infection)

        ax2.bar(x_positions[idx], avg_infection,
                yerr=std_infection, capsize=5, alpha=0.7,
                label=f'm = {m_consciousness}')

        # 在柱子上添加数值
        ax2.text(x_positions[idx], avg_infection + 0.01, f'{avg_infection:.3f}',
                 ha='center', va='bottom', fontsize=10)

    # 设置x轴刻度和标签
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'm = {m}' for m in m_consciousness_values])
    ax2.set_xlabel('m', fontsize=20)
    ax2.set_ylabel('$\\rho^{AI**}$', fontsize=20)
    # ax2.set_title('Impact of Consciousness Layer Connectivity on Infection',
    #               fontsize=13, fontweight='bold')
    ax2.set_ylim(0, y_max * 1.35)  # 压缩纵轴，给图例留空间
    ax2.legend(fontsize=14+7, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consciousness_impact_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'consciousness_impact_summary.pdf'),
                bbox_inches='tight')
    plt.show()


# ---------------------------
# 加载已有结果
# ---------------------------
def load_existing_results(output_dir):
    """加载已有结果"""
    results_matrices = {}

    for m_consciousness in m_consciousness_values:
        m_dir = os.path.join(output_dir, f"m{m_consciousness}")
        matrix_file = os.path.join(m_dir, "result_matrix.npy")

        if os.path.exists(matrix_file):
            matrix = np.load(matrix_file)
            results_matrices[m_consciousness] = matrix
            print(f"已加载 m={m_consciousness} 的结果")
        else:
            print(f"未找到 m={m_consciousness} 的结果文件")

    return results_matrices


# ---------------------------
# 分析结果
# ---------------------------
def analyze_results(results_matrices, output_dir):
    """分析实验结果"""
    print("\n=== 实验3结果分析 ===")

    analysis_results = {}

    for m_consciousness in m_consciousness_values:
        matrix = results_matrices[m_consciousness]
        valid_data = matrix[matrix >= 0]  # 排除错误值

        # 基本统计
        min_infection = np.min(valid_data)
        max_infection = np.max(valid_data)
        avg_infection = np.mean(valid_data)

        # 找到最优和最差参数组合
        flat_idx = np.argmin(valid_data)
        best_params_idx = np.unravel_index(flat_idx, matrix.shape)
        best_lambd = lambd_values[best_params_idx[0]]
        best_delta = delta_values[best_params_idx[1]]
        best_infection = valid_data[flat_idx]

        flat_idx_max = np.argmax(valid_data)
        worst_params_idx = np.unravel_index(flat_idx_max, matrix.shape)
        worst_lambd = lambd_values[worst_params_idx[0]]
        worst_delta = delta_values[worst_params_idx[1]]
        worst_infection = valid_data[flat_idx_max]

        analysis_results[m_consciousness] = {
            'min_infection': min_infection,
            'max_infection': max_infection,
            'avg_infection': avg_infection,
            'best_params': {'lambd': best_lambd, 'delta': best_delta, 'infection': best_infection},
            'worst_params': {'lambd': worst_lambd, 'delta': worst_delta, 'infection': worst_infection}
        }

        print(f"\n--- m = {m_consciousness} ---")
        print(f"  感染密度范围: {min_infection:.4f} - {max_infection:.4f}")
        print(f"  平均感染密度: {avg_infection:.4f}")
        print(f"  最优参数: λ={best_lambd}, δ={best_delta}, 感染密度={best_infection:.4f}")
        print(f"  最差参数: λ={worst_lambd}, δ={worst_delta}, 感染密度={worst_infection:.4f}")

    # 保存分析结果
    with open(os.path.join(output_dir, "analysis_results.json"), 'w') as f:
        json.dump(analysis_results, f, indent=2)

    return analysis_results


# ---------------------------
# 主函数
# ---------------------------
def main():
    output_dir = _os.path.join(_DATA, "results_experiment3")

    results_matrices = load_existing_results(output_dir)
    if len(results_matrices) != len(m_consciousness_values):
        print("部分结果缺失，请先运行完整的实验3")
        return

    print("成功加载所有结果，重新生成图片...")

    # 绘制热力图
    plot_heatmaps(results_matrices, output_dir)

    # 分析结果
    analyze_results(results_matrices, output_dir)

    print(f"\n=== 实验3图片重新生成完成 ===")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    if "--run" in _sys.argv:
        # Re-run the full simulations (heavy), then regenerate the figures.
        run_experiment3()
    main()
