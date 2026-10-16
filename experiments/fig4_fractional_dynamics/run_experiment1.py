# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_experiment1.py
import gc
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from fast_caputo_solver import solve_caputo_blocked, BlockConfig

# ---------------------------
# 参数配置
# ---------------------------
# 三组参数配置
parameter_sets = [
    {
        'name': 'Baseline',
        'lambd': 0.3,
        'beta_U': 0.6,
        'gamma': 1 / 3,
        'delta': 0.5,
        'mu': 1.0
    },
    {
        'name': 'High Infection',
        'lambd': 0.1,  # 降低意识传播
        'beta_U': 1.5,  # 大幅提高感染率
        'gamma': 1 / 3,
        'delta': 0.5,
        'mu': 1.0
    },
    {
        'name': 'High Awareness',
        'lambd': 1.0,  # 大幅提高意识传播
        'beta_U': 0.3,  # 降低感染率
        'gamma': 1 / 3,
        'delta': 0.2,  # 降低遗忘率，意识更持久
        'mu': 1.0
    }
]

# 三种不同的alpha值
alphas = [0.7, 0.8, 0.9]

# 固定参数
N = 250
m_consciousness = 4
m_risk = 2
T = 20.0
N_steps = 2048 * 2


# ---------------------------
# 初值生成
# ---------------------------
def generate_initial_conditions(N, dominant_state=0, dominance=0.8):
    y0 = np.zeros(3 * N)
    for i in range(N):
        p = np.zeros(3)
        p[dominant_state] = dominance
        remaining = 1.0 - dominance
        r1, r2 = np.random.rand(2)
        s = r1 + r2
        other = [j for j in range(3) if j != dominant_state]
        p[other] = remaining * np.array([r1, r2]) / s
        y0[3 * i:3 * i + 3] = p
    return y0


# ---------------------------
# 方程右端
# ---------------------------
def make_F(A, B, params):
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
# 计算群体平均值
# ---------------------------
def compute_population_averages(y, N):
    """计算U, A, I的群体平均值"""
    t_steps = y.shape[0]
    U_avg = np.zeros(t_steps)
    A_avg = np.zeros(t_steps)
    I_avg = np.zeros(t_steps)

    for t in range(t_steps):
        # 重塑为 (N, 3) 矩阵
        state_matrix = y[t].reshape(N, 3)
        U_avg[t] = np.mean(state_matrix[:, 0])  # U状态
        A_avg[t] = np.mean(state_matrix[:, 1])  # A状态
        I_avg[t] = np.mean(state_matrix[:, 2])  # I状态

    return U_avg, A_avg, I_avg


# ---------------------------
# 绘制结果（与整数阶相同规格）
# ---------------------------
def plot_results(results, output_dir):
    """绘制三组参数、三种alpha的结果，使用与整数阶相同的规格"""
    states = ['U (Unaware)', 'A (Aware)', 'I (Infected)']
    
    # 使用与整数阶实验相同的颜色方案
    colors = {
        'Baseline': '#1f77b4',      # 蓝色
        'High Infection': '#d62728', # 红色  
        'High Awareness': '#2ca02c'  # 绿色
    }
    
    # 三种alpha对应的线型
    linestyles = {
        0.7: '-',     # 实线
        0.8: '--',    # 虚线
        0.9: ':'      # 点线
    }
    
    # 线宽
    linewidth = 2.5
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for state_idx, state_name in enumerate(states):
        ax = axes[state_idx]
        
        # 设置对数横轴
        ax.set_xscale('log')
        
        for param_set in parameter_sets:
            param_name = param_set['name']
            color = colors[param_name]
            
            for alpha in alphas:
                key = f"{param_name}_alpha_{alpha}"
                if key in results:
                    t = results[key]['t']
                    
                    # 跳过t=0点以避免对数坐标问题
                    t_plot = t[1:]

                    if state_name == 'U (Unaware)':

                        state_data = results[key]['U_avg'][1:]
                    elif state_name == 'A (Aware)':

                        state_data = results[key]['A_avg'][1:]
                    else:  # 'I (Infected)'

                        state_data = results[key]['I_avg'][1:]
                    
                    linestyle = linestyles[alpha]
                    label = f"{param_name} (α={alpha})"
                    
                    ax.plot(t_plot, state_data, color=color, linestyle=linestyle, 
                           linewidth=linewidth, label=label)
        
        # 设置坐标轴标签和范围，与整数阶保持一致US状态密度
        ax.set_xlabel('Time', fontsize=16)
        if state_name == 'U (Unaware)':
            state_name = 'US'
        elif state_name == 'A (Aware)':
            state_name = 'AS'
        else:  # 'I (Infected)'
            state_name = 'AI'
        ax.set_ylabel(f'{state_name} state density', fontsize=16)
        ax.set_title(state_name, fontsize=17, fontweight='bold')
        if state_name == 'US':
            state_name = 'U (Unaware)'
        elif state_name == 'AS':
            state_name = 'A (Aware)'
        else:  # 'I (Infected)'
            state_name = 'I (Infected)'
        # 设置x轴范围（对数坐标）
        ax.set_xlim(0.01, T)
        
        # 设置y轴范围固定在0~1
        ax.set_ylim(0, 1.0)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 设置刻度
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # 添加图例（仅第一张子图）
        if state_idx == 0:
            ax.legend(fontsize=10, framealpha=0.9, loc='best')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, 'experiment1_fractional_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'experiment1_fractional_results.pdf'), 
                bbox_inches='tight')
    plt.show()

    # 额外创建一个汇总图，将所有状态放在一起比较
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    # 只绘制Baseline参数组，比较不同alpha的影响
    param_name = 'Baseline'
    for state_idx, state_name in enumerate(states):
        color = ['#1f77b4', '#ff7f0e', '#2ca02c'][state_idx]  # 蓝、橙、绿
        
        for alpha in alphas:
            key = f"{param_name}_alpha_{alpha}"
            if key in results:
                t = results[key]['t'][1:]
                
                if state_name == 'U (Unaware)':

                    state_data = results[key]['U_avg'][1:]
                    linestyle = linestyles[alpha]
                    label = f"US (α={alpha})"
                elif state_name == 'A (Aware)':

                    state_data = results[key]['A_avg'][1:]
                    linestyle = linestyles[alpha]
                    label = f"AS (α={alpha})"
                else:  # 'I (Infected)'

                    state_data = results[key]['I_avg'][1:]
                    linestyle = linestyles[alpha]
                    label = f"AI (α={alpha})"
                
                ax2.plot(t, state_data, color=color, linestyle=linestyle, 
                        linewidth=linewidth, label=label)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Time', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)
    ax2.set_title('Baseline Parameters: Effect of Fractional Order α', 
                 fontsize=17, fontweight='bold')
    ax2.set_xlim(0.01, T)
    ax2.set_ylim(0, 1.0)  # 确保y轴范围固定在0~1
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)
    ax2.legend(fontsize=14, framealpha=0.9, loc='upper right', ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_alpha_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'baseline_alpha_comparison.pdf'), 
                bbox_inches='tight')
    plt.show()


# ---------------------------
# 主运行函数
# ---------------------------
def run_experiment1(output_dir=_os.path.join(_DATA, "results_experiment1")):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # 固定随机种子以保证可重复性
    np.random.seed(42)

    # 生成网络（所有参数集共享相同的网络结构）
    Gc = nx.barabasi_albert_graph(N, m_consciousness, seed=42)
    Gr = nx.barabasi_albert_graph(N, m_risk, seed=43)
    A, B = nx.to_numpy_array(Gc), nx.to_numpy_array(Gr)

    # 生成初值
    y0 = generate_initial_conditions(N)

    # 对每组参数和每个alpha值进行求解
    for param_set in parameter_sets:
        param_name = param_set['name']

        for alpha in alphas:
            case_name = f"{param_name}_alpha_{alpha}"
            case_dir = os.path.join(output_dir, case_name)
            result_file = os.path.join(case_dir, "results.npz")

            if os.path.exists(result_file):
                print(f"Loading existing results for {case_name}")
                # 加载已有结果
                data = np.load(result_file)
                t = data['t']
                y = data['y']
                U_avg, A_avg, I_avg = data['U_avg'], data['A_avg'], data['I_avg']
            else:
                print(f"\n=== Running {case_name} ===")

                # 准备参数
                params = dict(
                    N=N, mu=param_set['mu'], delta=param_set['delta'],
                    lambd=param_set['lambd'], beta_U=param_set['beta_U'],
                    gamma=param_set['gamma'], alpha=alpha,
                    m_consciousness=m_consciousness, m_risk=m_risk,
                    T=T, N_steps=N_steps
                )

                # 创建方程右端函数
                F = make_F(A, B, params)

                # 配置求解器
                cfg = BlockConfig(16, 4, 4096, tol=1e-4, verbose=True,
                                  picard_omega=0.7, omega_backtrack=0.8)

                # 求解分数阶方程
                t0 = time.time()
                t, y = solve_caputo_blocked(F, alpha, y0, T, N_steps, cfg)
                print(f"Time {time.time() - t0:.2f}s")

                # 计算群体平均值
                U_avg, A_avg, I_avg = compute_population_averages(y, N)

                # 保存结果
                os.makedirs(case_dir, exist_ok=True)
                np.savez(result_file, t=t, y=y, U_avg=U_avg, A_avg=A_avg, I_avg=I_avg)

                # 保存参数
                with open(os.path.join(case_dir, "params.json"), "w") as f:
                    json.dump(params, f, indent=2)

                # 清理内存
                del F, y
                gc.collect()

            # 存储结果用于绘图
            results[case_name] = {
                't': t,
                'U_avg': U_avg,
                'A_avg': A_avg,
                'I_avg': I_avg,
                'params': param_set,
                'alpha': alpha
            }

    # 绘制结果
    print("\n=== Plotting results ===")
    plot_results(results, output_dir)

    # 保存汇总结果
    summary_file = os.path.join(output_dir, "experiment1_summary.json")
    summary_data = {}
    for key, result in results.items():
        summary_data[key] = {
            'final_U': float(result['U_avg'][-1]),
            'final_A': float(result['A_avg'][-1]),
            'final_I': float(result['I_avg'][-1]),
            'params': result['params'],
            'alpha': result['alpha']
        }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n=== Experiment 1 completed ===")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    run_experiment1()