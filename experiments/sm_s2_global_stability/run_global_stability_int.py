# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_global_stability_int.py
# 全局稳定性数值验证实验（整数阶主实验）
# 12 种不同初值 → 检查收敛到同一稳态
import os, json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from integer_order_solver import UAU_SIS_IntegerSolver, NetworkType

N = 250
m_consciousness, m_risk = 4, 2
SEED = 42
T = 20.0
LAM, DELTA, MU, GAMMA = 0.3, 0.5, 1.0, 1/3

np.random.seed(SEED)
Gc = nx.barabasi_albert_graph(N, m_consciousness, seed=SEED)
Gr = nx.barabasi_albert_graph(N, m_risk, seed=SEED+1)
A = nx.to_numpy_array(Gc)
B = nx.to_numpy_array(Gr)

def make_ic(inf_frac, awr_frac, seed):
    rng = np.random.default_rng(seed)
    y0 = np.zeros(3*N)
    for i in range(N):
        p = np.array([1-inf_frac-awr_frac, awr_frac, inf_frac])
        p += rng.normal(0, 0.02, 3)
        p = np.clip(p, 0, 1)
        p = p / p.sum()
        y0[3*i:3*i+3] = p
    return y0

IC_LIST = [
    (0.01, 0.01), (0.01, 0.30), (0.01, 0.60),
    (0.05, 0.01), (0.05, 0.30), (0.05, 0.60),
    (0.15, 0.01), (0.15, 0.30), (0.15, 0.60),
    (0.30, 0.01), (0.30, 0.30), (0.30, 0.60),
]

def run_integer(betaU, inf_frac, awr_frac, seed):
    solver = UAU_SIS_IntegerSolver(
        N=N, network_type_consciousness=NetworkType.BA, network_type_risk=NetworkType.BA,
        network_params_consciousness={'m': m_consciousness}, network_params_risk={'m': m_risk}, seed=SEED)
    solver.set_custom_networks(A, B)
    solver.set_parameters(lambd=LAM, beta_U=betaU, gamma=GAMMA, delta=DELTA, mu=MU)
    y0 = make_ic(inf_frac, awr_frac, seed)
    res = solver.solve(t_span=(0, T), t_eval=np.linspace(0, T, 401),
                       initial_conditions=y0, show_progress=False)
    return res['t'], res['rho_AI'], res['rho_AI'][-1]

results = {}
for betaU, tag in [(0.6, 'endemic'), (0.1, 'free')]:
    results[tag] = {'trajs': [], 'finals': []}
    print(f"=== integer-order beta^U={betaU} ({tag}) ===", flush=True)
    for idx, (i_, a_) in enumerate(IC_LIST):
        t, rho, fin = run_integer(betaU, i_, a_, 1000+idx)
        results[tag]['trajs'].append((i_, a_, t, rho))
        results[tag]['finals'].append(fin)
        print(f"  IC({i_:.2f},{a_:.2f}) -> final rho^AI = {fin:.5f}", flush=True)
    finals = np.array(results[tag]['finals'])
    print(f"  spread: min={finals.min():.5f}, max={finals.max():.5f}, range={finals.max()-finals.min():.6f}", flush=True)

plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
colors = plt.cm.viridis(np.linspace(0, 1, len(IC_LIST)))
for tag, ax, title, ymax in [
    ('endemic', axes[0], r'$\mathcal{R}_0>1$: convergence to the rumor-endemic equilibrium', 0.55),
    ('free', axes[1], r'$\mathcal{R}_0<1$: convergence to the rumor-free equilibrium', 0.05)]:
    for idx, (i_, a_, t, rho) in enumerate(results[tag]['trajs']):
        ax.plot(t, rho, color=colors[idx], lw=1.5, alpha=0.85)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\rho^{AI}$')
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.01, ymax)
    if tag == 'endemic':
        f0 = results['endemic']['finals'][0]
        ax.axhline(y=f0, color='gray', ls='--', lw=1)
        ax.text(0.5, f0+0.015, r'$\rho^{AI**}$', color='gray')

plt.tight_layout()
plt.savefig(_os.path.join(_DATA, 'global_stability', 'global_stability_verification.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(_os.path.join(_DATA, 'global_stability', 'global_stability_verification.png'), dpi=300, bbox_inches='tight')
print("\nFigure saved: global_stability_verification.pdf", flush=True)

summary = {}
for tag in ['endemic', 'free']:
    finals = results[tag]['finals']
    summary[tag] = {'mean': float(np.mean(finals)), 'min': float(min(finals)),
                    'max': float(max(finals)), 'range': float(max(finals)-min(finals))}
with open(_os.path.join(_DATA, 'global_stability', 'global_stability_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary:", summary, flush=True)
