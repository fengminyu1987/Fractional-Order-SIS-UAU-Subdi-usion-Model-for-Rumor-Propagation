# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_experiment4.py
import gc
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

from fast_caputo_solver import solve_caputo_blocked, BlockConfig

# ---------------------------
# 参数配置
# ---------------------------
# 固定参数
N = 500
m_consciousness = 4  # 信息层平均度
m_risk = 2           # 风险层平均度
T = 20.0
N_steps = 2048
alpha = 0.8  # 固定alpha值

# 固定模型参数
params_fixed = {
    'lambd': 0.1,    # 意识传播率
    'beta_U': 1.5,   # 未意识者感染率
    'gamma': 1/3,    # 意识者感染降低因子
    'delta': 0.5,    # 遗忘率
    'mu': 1.0        # 恢复率
}

# 网络类型
network_types = ['ER', 'BA', 'Regular']

# 状态颜色配置
state_colors = {
    'US': 'green',   # 未意识易感
    'AS': 'blue',    # 意识易感  
    'AI': 'red'      # 意识感染
}

# ---------------------------
# 网络生成函数
# ---------------------------
def generate_network(net_type, N, avg_degree, layer_name, seed_base=42):
    """生成指定类型的网络"""
    seed = seed_base + hash(net_type + layer_name) % 1000
    
    if net_type == 'ER':
        p = avg_degree / (N - 1)
        G = nx.erdos_renyi_graph(N, p, seed=seed)
    elif net_type == 'BA':
        m = avg_degree // 2
        G = nx.barabasi_albert_graph(N, m, seed=seed)
    elif net_type == 'Regular':
        G = nx.random_regular_graph(avg_degree, N, seed=seed)
    else:
        raise ValueError(f"Unknown network type: {net_type}")
    
    return nx.to_numpy_array(G)

def generate_networks_preset():
    """预先生成所有需要的网络"""
    networks = {}
    
    # 信息层网络 (平均度 = 4)
    for net_type in network_types:
        key = f"consciousness_{net_type}"
        networks[key] = generate_network(net_type, N, m_consciousness, "consciousness")
        print(f"Generated {key}, actual avg degree: {np.mean(np.sum(networks[key], axis=1)):.2f}")
    
    # 风险层网络 (平均度 = 2)  
    for net_type in network_types:
        key = f"risk_{net_type}"
        networks[key] = generate_network(net_type, N, m_risk, "risk")
        print(f"Generated {key}, actual avg degree: {np.mean(np.sum(networks[key], axis=1)):.2f}")
    
    return networks

# ---------------------------
# 初值生成
# ---------------------------
def generate_initial_conditions(N, dominant_state=0, dominance=0.8):
    """生成初始条件"""
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
    """创建方程右端函数"""
    N = params["N"]
    beta_U, beta_A = params["beta_U"], params["beta_U"] * params["gamma"]
    lambd, delta, mu = params["lambd"], params["delta"], params["mu"]

    def F(P, t):
        M = len(P) - 1
        P_data = P[:-1, :].reshape(M, N, 3)
        BP, AP = np.zeros_like(P_data), np.zeros_like(P_data)
        for j in range(M):
            BP[j] = A @ P_data[j]  # 信息层传播
            AP[j] = B @ P_data[j]  # 风险层传播
        
        M3 = np.array([[0, 0, 0],
                      [delta, -delta, 0],
                      [0, mu, -mu]])
        
        Q = np.zeros((N, M, 3, 3))
        for i in range(N):
            e = np.zeros(N); e[i] = 1
            for j in range(M):
                s1 = e @ BP[j] @ np.array([[0], [1], [1]])  # 信息层影响
                s2 = e @ AP[j] @ np.array([[0], [0], [1]])  # 风险层影响
                
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
    """计算US, AS, AI的群体平均值"""
    t_steps = y.shape[0]
    US_avg = np.zeros(t_steps)  # 未意识易感
    AS_avg = np.zeros(t_steps)  # 意识易感  
    AI_avg = np.zeros(t_steps)  # 意识感染

    for t in range(t_steps):
        state_matrix = y[t].reshape(N, 3)
        US_avg[t] = np.mean(state_matrix[:, 0])  # US状态
        AS_avg[t] = np.mean(state_matrix[:, 1])  # AS状态  
        AI_avg[t] = np.mean(state_matrix[:, 2])  # AI状态

    return US_avg, AS_avg, AI_avg

# ---------------------------
# 绘制改进的3x3网格结果
# ---------------------------
def plot_enhanced_network_comparison(results, output_dir):
    """绘制改进的3x3网络比较图"""
    fig = plt.figure(figsize=(16, 12))
    
    # 使用GridSpec创建复杂的子图布局
    gs = GridSpec(4, 4, figure=fig, height_ratios=[0.1, 1, 1, 1], width_ratios=[0.1, 1, 1, 1],
                  hspace=0.3, wspace=0.3)
    
    # 创建3x3的主网格
    main_axes = []
    for i in range(3):
        row_axes = []
        for j in range(3):
            # 每个格子包含上下两个子图
            cell_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i+1, j+1], 
                                            height_ratios=[1, 0.6], hspace=0.1)
            ax_top = fig.add_subplot(cell_gs[0])  # 上半部分：时间序列
            ax_bottom = fig.add_subplot(cell_gs[1])  # 下半部分：柱状图
            row_axes.append((ax_top, ax_bottom))
        main_axes.append(row_axes)
    
    # 添加坐标轴标签
    # 纵轴标签（信息层类型）
    for i, net_type in enumerate(network_types):
        ax_label = fig.add_subplot(gs[i+1, 0])
        ax_label.text(0.5, 0.5, net_type, rotation=90, ha='center', va='center', 
                     fontsize=18, fontweight='bold', transform=ax_label.transAxes)
        ax_label.axis('off')
    
    # 横轴标签（风险层类型）
    for j, net_type in enumerate(network_types):
        ax_label = fig.add_subplot(gs[0, j+1])
        ax_label.text(0.5, 0.5, net_type, ha='center', va='center', 
                     fontsize=18, fontweight='bold', transform=ax_label.transAxes)
        ax_label.axis('off')
    
    # 收集所有最终AI密度用于颜色映射
    final_AI_densities = []
    for i, conscious_type in enumerate(network_types):
        for j, risk_type in enumerate(network_types):
            key = f"{conscious_type}_{risk_type}"
            if key in results:
                final_AI_densities.append(results[key]['final_AI'])
    
    if not final_AI_densities:
        print("No results to plot!")
        return
        
    vmin, vmax = min(final_AI_densities), max(final_AI_densities)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    
    # 绘制每个格子
    for i, conscious_type in enumerate(network_types):
        for j, risk_type in enumerate(network_types):
            key = f"{conscious_type}_{risk_type}"
            ax_top, ax_bottom = main_axes[i][j]
            
            if key in results:
                t = results[key]['t']
                US_avg = results[key]['US_avg']
                AS_avg = results[key]['AS_avg'] 
                AI_avg = results[key]['AI_avg']
                final_US = results[key]['final_US']
                final_AS = results[key]['final_AS']
                final_AI = results[key]['final_AI']
                
                # 根据最终AI密度选择颜色
                cell_color = cmap(norm(final_AI))
                
                # 创建带有透明度的背景色
                bg_color = list(cell_color)
                bg_color[3] = 0.4  # 设置alpha通道为0.1
                
                # 上半部分：时间序列图（压扁高度）
                # 使用对数x轴
                ax_top.semilogx(t[1:], US_avg[1:], color=state_colors['US'], 
                              linewidth=1.5, label='US')
                ax_top.semilogx(t[1:], AS_avg[1:], color=state_colors['AS'], 
                              linewidth=1.5, label='AS')
                ax_top.semilogx(t[1:], AI_avg[1:], color=state_colors['AI'], 
                              linewidth=1.5, label='AI')
                
                # 设置y轴范围并添加网格
                ax_top.set_ylim(0, 1)
                ax_top.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax_top.set_xticklabels([])  # 移除x轴刻度标签
                
                # 下半部分：最终状态柱状图
                states = ['US', 'AS', 'AI']
                final_values = [final_US, final_AS, final_AI]
                colors = [state_colors[s] for s in states]
                
                bars = ax_bottom.bar(states, final_values, color=colors, alpha=0.8)
                
                # 在柱子上添加数值标签
                for bar, value in zip(bars, final_values):
                    height = bar.get_height()
                    ax_bottom.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                 f'{value:.2f}', ha='center', va='bottom', fontsize=12)
                
                ax_bottom.set_ylim(0, 1)
                ax_bottom.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax_top.tick_params(labelsize=11)
                ax_bottom.tick_params(labelsize=11)
                
                # 设置格子背景色反映最终AI密度
                ax_top.set_facecolor(bg_color)
                ax_bottom.set_facecolor(bg_color)
                
                # 只在边缘格子添加标签
                if i == 2:  # 最后一行
                    ax_bottom.set_xlabel('State', fontsize=13)
                if j == 0:  # 第一列
                    ax_top.set_ylabel('Fraction', fontsize=13)
                    ax_bottom.set_ylabel('Final Density', fontsize=13)
            
            else:
                ax_top.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax_top.transAxes, fontsize=12)
                ax_bottom.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                             transform=ax_bottom.transAxes, fontsize=12)
    
    # 添加全局图例
    legend_ax = fig.add_subplot(gs[0, 0])
    legend_ax.axis('off')
    legend_elements = [
        Patch(facecolor=state_colors['US'], label='US (Unaware Susceptible)'),
        Patch(facecolor=state_colors['AS'], label='AS (Aware Susceptible)'),
        Patch(facecolor=state_colors['AI'], label='AI (Aware Infected)')
    ]
    legend_ax.legend(handles=legend_elements, loc='center', frameon=False, 
                    fontsize=14, bbox_to_anchor=(0.5, 0.5))
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Final AI Density', rotation=270, labelpad=20, fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # # 添加标题
    # plt.suptitle('Disease Spread on Different Network Structures\n(Information Layer × Risk Layer)',
    #              fontsize=16, fontweight='bold', y=0.95)
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, 'experiment4_enhanced_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'experiment4_enhanced_comparison.pdf'), 
                bbox_inches='tight')
    plt.show()

# ---------------------------
# 主运行函数
# ---------------------------
def run_experiment4(output_dir=_os.path.join(_DATA, "results_experiment4")):
    """运行实验4：不同网络结构对传播的影响"""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # 固定随机种子
    np.random.seed(42)
    
    print("=== Generating Networks ===")
    networks = generate_networks_preset()
    
    # 生成初值
    y0 = generate_initial_conditions(N)
    
    print("\n=== Running Simulations ===")
    
    # 运行所有网络组合
    for i, conscious_type in enumerate(network_types):
        for j, risk_type in enumerate(network_types):
            key = f"{conscious_type}_{risk_type}"
            case_dir = os.path.join(output_dir, key)
            result_file = os.path.join(case_dir, "results.npz")
            
            if os.path.exists(result_file):
                print(f"Loading existing results for {key}")
                # 加载已有结果
                data = np.load(result_file)
                t = data['t']
                y = data['y']
                US_avg, AS_avg, AI_avg = data['US_avg'], data['AS_avg'], data['AI_avg']
            else:
                print(f"\n--- Running {key} ---")
                
                # 获取网络
                A = networks[f"consciousness_{conscious_type}"]  # 信息层
                B = networks[f"risk_{risk_type}"]               # 风险层
                
                # 准备参数
                params = {
                    'N': N,
                    'mu': params_fixed['mu'],
                    'delta': params_fixed['delta'],
                    'lambd': params_fixed['lambd'],
                    'beta_U': params_fixed['beta_U'],
                    'gamma': params_fixed['gamma'],
                    'alpha': alpha,
                    'm_consciousness': m_consciousness,
                    'm_risk': m_risk,
                    'T': T,
                    'N_steps': N_steps
                }
                
                # 创建方程右端函数
                F = make_F(A, B, params)
                
                # 配置求解器（使用更保守的参数确保收敛）
                cfg = BlockConfig(
                    block_steps=512, 
                    min_block_steps=64, 
                    max_block_steps=2048,
                    grow_factor=1.4, 
                    shrink_factor=0.6, 
                    target_iter=8,
                    max_iter=100, 
                    tol=1e-5, 
                    verbose=False,
                    picard_omega=0.6, 
                    omega_backtrack=0.7
                )
                
                # 求解分数阶方程
                t0 = time.time()
                try:
                    t, y = solve_caputo_blocked(F, alpha, y0, T, N_steps, cfg)
                    print(f"Success! Time: {time.time() - t0:.2f}s")
                except Exception as e:
                    print(f"Failed: {e}")
                    continue
                
                # 计算群体平均值
                US_avg, AS_avg, AI_avg = compute_population_averages(y, N)
                
                # 保存结果
                os.makedirs(case_dir, exist_ok=True)
                np.savez(result_file, t=t, y=y, US_avg=US_avg, AS_avg=AS_avg, AI_avg=AI_avg)
                
                # 保存参数和网络信息
                case_params = params.copy()
                case_params.update({
                    'consciousness_network': conscious_type,
                    'risk_network': risk_type
                })
                with open(os.path.join(case_dir, "params.json"), "w") as f:
                    json.dump(case_params, f, indent=2)
                
                # 清理内存
                del F, y
                gc.collect()
            
            # 存储结果用于绘图（使用正确的状态名称）
            results[key] = {
                't': t,
                'US_avg': US_avg,
                'AS_avg': AS_avg, 
                'AI_avg': AI_avg,
                'final_US': US_avg[-1],
                'final_AS': AS_avg[-1],
                'final_AI': AI_avg[-1],
                'consciousness_network': conscious_type,
                'risk_network': risk_type
            }
    
    # 绘制结果
    print("\n=== Plotting Results ===")
    plot_enhanced_network_comparison(results, output_dir)
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, "experiment4_summary.json")
    summary_data = {}
    for key, result in results.items():
        summary_data[key] = {
            'final_US': float(result['final_US']),
            'final_AS': float(result['final_AS']),
            'final_AI': float(result['final_AI']),
            'consciousness_network': result['consciousness_network'],
            'risk_network': result['risk_network']
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # 打印统计信息
    print("\n=== Experiment 4 Summary ===")
    print("Final State Densities:")
    for conscious_type in network_types:
        for risk_type in network_types:
            key = f"{conscious_type}_{risk_type}"
            if key in results:
                print(f"  {key}: US={results[key]['final_US']:.3f}, "
                      f"AS={results[key]['final_AS']:.3f}, AI={results[key]['final_AI']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    run_experiment4()