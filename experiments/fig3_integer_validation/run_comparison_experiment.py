# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_comparison_experiment.py
"""
实验1综合比较：同时运行IBMFA和蒙特卡洛模拟
对比三组不同参数下的状态演化
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm

# 导入求解器
from integer_order_solver import UAU_SIS_IntegerSolver, NetworkType
from monte_carlo_simulator import UAU_SIS_MonteCarlo


def run_experiment_comparison():
    """运行综合比较实验"""

    # 三组更极端的参数配置，确保结果有明显差异
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

    results = {}

    for i, params in enumerate(tqdm(parameter_sets, desc="参数组")):
        print(f"\n=== 运行参数组: {params['name']} ===")
        print(f"参数: λ={params['lambd']}, β_U={params['beta_U']}, "
              f"γ={params['gamma']}, δ={params['delta']}, μ={params['mu']}")

        # 运行IBMFA
        print("运行IBMFA求解器...")
        solver = UAU_SIS_IntegerSolver(
            N=250,  # 使用较小网络以加快蒙特卡洛模拟
            network_type_consciousness=NetworkType.BA,
            network_type_risk=NetworkType.BA,
            network_params_consciousness={'m': 4},
            network_params_risk={'m': 2},
            seed=42 + i
        )

        solver.set_parameters(**params)

        ibmfa_result = solver.solve(
            t_span=(0, 10.0),  # 延长模拟时间以观察更多动态
            t_eval=np.linspace(0, 10.0, 251),
            initial_conditions_kwargs={
                'initial_state': 'mixed',
                'infected_fraction': 0.02,
                'aware_fraction': 0.03
            },
            show_progress=False
        )

        # 运行蒙特卡洛模拟
        print("运行蒙特卡洛模拟...")
        mc_simulator = UAU_SIS_MonteCarlo(
            N=250,
            network_type_consciousness="BA",
            network_type_risk="BA",
            network_params_consciousness={'m': 4},
            network_params_risk={'m': 2},
            seed=42 + i
        )

        mc_simulator.set_parameters(**params)

        mc_result = mc_simulator.run_multiple_simulations(
            n_simulations=15,  # 减少模拟次数以加快速度
            T=10.0,
            record_interval=0.04,
            initial_infected=0.02,
            initial_aware=0.03
        )

        results[params['name']] = {
            'params': params.copy(),
            'ibmfa': ibmfa_result,
            'monte_carlo': mc_result
        }

        print(f"IBMFA最终状态: US={ibmfa_result['rho_US'][-1]:.4f}, "
              f"AS={ibmfa_result['rho_AS'][-1]:.4f}, AI={ibmfa_result['rho_AI'][-1]:.4f}")
        print(f"MC最终状态: US={mc_result['mean_densities']['US'][-1]:.4f}, "
              f"AS={mc_result['mean_densities']['AS'][-1]:.4f}, AI={mc_result['mean_densities']['AI'][-1]:.4f}")

    return results


def plot_comprehensive_comparison(results, save_path=_os.path.join(_FIGD, 'comprehensive_comparison.pdf')):
    """绘制综合比较图"""

    # 设置matplotlib样式
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12
    })

    # 设置颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色，橙色，绿色
    linestyles_ibmfa = ['-', '--', '-.']
    linestyles_mc = [':', ':', ':']  # 蒙特卡洛使用虚线

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 状态名称和标签
    states = ['US', 'AS', 'AI']
    state_labels = [r'$\rho^{US**}$', r'$\rho^{AS**}$', r'$\rho^{AI**}$']

    for state_idx, state in enumerate(states):
        ax = axes[state_idx]

        for i, (param_name, data) in enumerate(results.items()):
            color = colors[i]

            # IBMFA结果
            ibmfa_time = data['ibmfa']['t'][1:]  # 跳过t=0
            ibmfa_density = data['ibmfa'][f'rho_{state}'][1:]

            ax.semilogx(ibmfa_time, ibmfa_density,
                        color=color, linestyle=linestyles_ibmfa[i], linewidth=2.5,
                        label=f'IBMFA {param_name}')

            # 蒙特卡洛结果
            mc_time = data['monte_carlo']['time_points'][1:]  # 跳过t=0
            mc_mean = data['monte_carlo']['mean_densities'][state][1:]
            mc_std = np.sqrt(data['monte_carlo']['variance_densities'][state][1:])

            ax.semilogx(mc_time, mc_mean,
                        color=color, linestyle=linestyles_mc[i], linewidth=2,
                        label=f'MC {param_name}')

            # 绘制方差带
            ax.fill_between(mc_time, mc_mean - mc_std, mc_mean + mc_std,
                            color=color, alpha=0.2)

        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel(state_labels[state_idx], fontsize=18)
        if state_idx == 0:
            ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_individual_states_comparison(results, save_dir=_os.path.join(_DATA, 'individual_plots_fig3')):
    """为每个状态单独绘制比较图"""

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 设置matplotlib样式
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12
    })

    # 设置颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色，橙色，绿色
    linestyles_ibmfa = ['-', '--', '-.']
    linestyles_mc = [':', ':', ':']  # 蒙特卡洛使用虚线

    states = ['US', 'AS', 'AI']
    state_labels = [r'$\rho^{US**}$', r'$\rho^{AS**}$', r'$\rho^{AI**}$']

    for state_idx, state in enumerate(states):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for i, (param_name, data) in enumerate(results.items()):
            color = colors[i]

            # IBMFA结果
            ibmfa_time = data['ibmfa']['t'][1:]  # 跳过t=0
            ibmfa_density = data['ibmfa'][f'rho_{state}'][1:]

            ax.semilogx(ibmfa_time, ibmfa_density,
                        color=color, linestyle=linestyles_ibmfa[i], linewidth=2.5,
                        label=f'IBMFA {param_name}')

            # 蒙特卡洛结果
            mc_time = data['monte_carlo']['time_points'][1:]  # 跳过t=0
            mc_mean = data['monte_carlo']['mean_densities'][state][1:]
            mc_std = np.sqrt(data['monte_carlo']['variance_densities'][state][1:])

            ax.semilogx(mc_time, mc_mean,
                        color=color, linestyle=linestyles_mc[i], linewidth=2,
                        label=f'MC {param_name}')

            # 绘制方差带
            ax.fill_between(mc_time, mc_mean - mc_std, mc_mean + mc_std,
                            color=color, alpha=0.2)

        ax.set_xlabel('Time', fontsize=16)
        ax.set_ylabel(state_labels[state_idx], fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=13)
        ax.set_title(f'State {state} Comparison')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/state_{state}_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()


def plot_parameter_analysis(results, save_path=_os.path.join(_FIGD, 'parameter_analysis.pdf')):
    """绘制参数影响分析图"""

    # 设置matplotlib样式
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12
    })

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 提取最终状态值
    param_names = list(results.keys())
    final_states_ibmfa = {
        'US': [results[name]['ibmfa']['rho_US'][-1] for name in param_names],
        'AS': [results[name]['ibmfa']['rho_AS'][-1] for name in param_names],
        'AI': [results[name]['ibmfa']['rho_AI'][-1] for name in param_names]
    }

    final_states_mc = {
        'US': [results[name]['monte_carlo']['mean_densities']['US'][-1] for name in param_names],
        'AS': [results[name]['monte_carlo']['mean_densities']['AS'][-1] for name in param_names],
        'AI': [results[name]['monte_carlo']['mean_densities']['AI'][-1] for name in param_names]
    }

    # 状态名称和标签
    states = ['US', 'AS', 'AI']
    state_labels = [r'$\rho^{US**}$', r'$\rho^{AS**}$', r'$\rho^{AI**}$']

    x_pos = np.arange(len(param_names))
    width = 0.35

    # 绘制最终状态条形图
    for i, state in enumerate(states):
        ax = axes[0, i]

        # IBMFA条形
        bars1 = ax.bar(x_pos - width / 2, final_states_ibmfa[state], width,
                       label='IBMFA', alpha=0.8, color='#1f77b4')

        # 蒙特卡洛条形
        bars2 = ax.bar(x_pos + width / 2, final_states_mc[state], width,
                       label='Monte Carlo', alpha=0.8, color='#ff7f0e')

        ax.set_xlabel('Parameter Set', fontsize=16)
        ax.set_ylabel(state_labels[i], fontsize=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_names, fontsize=14)
        if i == 0:
            ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=12)

        # 在条形上添加数值
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')

    # 绘制参数对比
    param_values = {
        'lambda': [results[name]['params']['lambd'] for name in param_names],
        'beta_U': [results[name]['params']['beta_U'] for name in param_names],
        'delta': [results[name]['params']['delta'] for name in param_names]
    }

    param_labels = ['$\lambda$', '$\\beta^U$', '$\delta$']

    for i, (param, values) in enumerate(param_values.items()):
        ax = axes[1, i]

        bars = ax.bar(x_pos, values, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Parameter Set', fontsize=16)
        ax.set_ylabel(param_labels[i], fontsize=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_names, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=12)

        # 在条形上添加数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.01,
                    f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_comparison_results(results, filename=_os.path.join(_DATA, 'comparison_results.pkl')):
    """保存比较结果"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Comparison results saved to {filename}")


def load_comparison_results(filename=_os.path.join(_DATA, 'comparison_results.pkl')):
    """加载比较结果"""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    print(f"Comparison results loaded from {filename}")
    return results


def analyze_comparison_results(results):
    """分析比较结果"""
    print("\n=== 综合比较结果分析 ===")

    for param_name, data in results.items():
        params = data['params']
        ibmfa = data['ibmfa']
        mc = data['monte_carlo']

        print(f"\n--- {param_name} ---")
        print(f"参数: λ={params['lambd']}, β_U={params['beta_U']}, "
              f"γ={params['gamma']}, δ={params['delta']}, μ={params['mu']}")

        # IBMFA结果
        ibmfa_final_US = ibmfa['rho_US'][-1]
        ibmfa_final_AS = ibmfa['rho_AS'][-1]
        ibmfa_final_AI = ibmfa['rho_AI'][-1]

        # 蒙特卡洛结果
        mc_final_US = mc['mean_densities']['US'][-1]
        mc_final_AS = mc['mean_densities']['AS'][-1]
        mc_final_AI = mc['mean_densities']['AI'][-1]

        print(f"IBMFA最终状态: US={ibmfa_final_US:.4f}, AS={ibmfa_final_AS:.4f}, AI={ibmfa_final_AI:.4f}")
        print(f"MC最终状态:    US={mc_final_US:.4f}, AS={mc_final_AS:.4f}, AI={mc_final_AI:.4f}")

        # 计算差异
        diff_US = abs(ibmfa_final_US - mc_final_US)
        diff_AS = abs(ibmfa_final_AS - mc_final_AS)
        diff_AI = abs(ibmfa_final_AI - mc_final_AI)

        print(f"差异: ΔUS={diff_US:.4f}, ΔAS={diff_AS:.4f}, ΔAI={diff_AI:.4f}")

        # 计算峰值
        ibmfa_peak_AI = np.max(ibmfa['rho_AI'])
        ibmfa_peak_time = ibmfa['t'][np.argmax(ibmfa['rho_AI'])]

        mc_peak_AI = np.max(mc['mean_densities']['AI'])
        mc_peak_time = mc['time_points'][np.argmax(mc['mean_densities']['AI'])]

        print(f"IBMFA感染峰值: {ibmfa_peak_AI:.4f} (时间: {ibmfa_peak_time:.2f})")
        print(f"MC感染峰值:    {mc_peak_AI:.4f} (时间: {mc_peak_time:.2f})")


if __name__ == "__main__":
    # 加载已有结果并重新生成图片
    if not os.path.exists(_os.path.join(_DATA, 'comparison_results.pkl')):
        print("错误：未找到 comparison_results.pkl，请先运行实验。")
        exit(1)

    results = load_comparison_results()
    print("成功加载已保存结果，重新生成图片...\n")

    # 绘制综合比较图
    plot_comprehensive_comparison(results)

    # 绘制单独状态比较图
    plot_individual_states_comparison(results)

    # 绘制参数分析图
    plot_parameter_analysis(results)

    # 分析结果
    analyze_comparison_results(results)
    print("\n图片重新生成完成。")