# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_global_stability.py
# 全局稳定性数值验证实验：
#   从大量不同初值出发（不同感染密度/意识密度分布），
#   验证系统是否收敛到同一稳态（R0>1 -> REE；R0<1 -> RFE）
#   整数阶跑全部初值（快），分数阶跑代表性初值（验证 α 无关性）
import os, json, time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from integer_order_solver import UAU_SIS_IntegerSolver, NetworkType
from fast_caputo_solver import solve_caputo_blocked, BlockConfig

N = 250
m_consciousness, m_risk = 4, 2
SEED = 42
T = 20.0

LAM, DELTA, MU, GAMMA = 0.3, 0.5, 1.0, 1/3

# 网络（与论文一致）
np.random.seed(SEED)
Gc = nx.barabasi_albert_graph(N, m_consciousness, seed=SEED)
Gr = nx.barabasi_albert_graph(N, m_risk, seed=SEED+1)
A = nx.to_numpy_array(Gc)
B = nx.to_numpy_array(Gr)

def make_ic(inf_frac, awr_frac, seed):
    """初值：US/AS/AI 按给定比例 + 随机扰动"""
    rng = np.random.default_rng(seed)
    y0 = np.zeros(3*N)
    for i in range(N):
        p = np.array([1-inf_frac-awr_frac, awr_frac, inf_frac])
        p += rng.normal(0, 0.02, 3)
        p = np.clip(p, 0, 1)
        p = p / p.sum()
        y0[3*i:3*i+3] = p
    return y0

# 初值集合：覆盖不同感染/意识水平
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

def run_fractional(betaU, alpha, inf_frac, awr_frac, seed):
    params = dict(N=N, mu=MU, delta=DELTA, lambd=LAM, beta_U=betaU, gamma=GAMMA,
                  alpha=alpha, m_consciousness=m_consciousness, m_risk=m_risk, T=T, N_steps=2048)
    from run_bifurcation import make_F  # 复用
    return None

# ---------- 主流程 ----------
results = {}

# 整数阶：全部 12 初值 × 2 参数组
for betaU, tag in [(0.6, 'endemic'), (0.1, 'free')]:
    results[tag] = {'t': None, 'trajs': [], 'finals': []}
    print(f"=== integer-order beta^U={betaU} ({tag}) ===")
    for idx, (i_, a_) in enumerate(IC_LIST):
        t, rho, fin = run_integer(betaU, i_, a_, 1000+idx)
        results[tag]['trajs'].append((i_, a_, t, rho))
        results[tag]['finals'].append(fin)
        print(f"  IC({i_:.2f},{a_:.2f}) -> final rho^AI = {fin:.5f}")
    finals = np.array(results[tag]['finals'])
    print(f"  spread: min={finals.min():.5f}, max={finals.max():.5f}, range={finals.max()-finals.min():.6f}")

# 分数阶：α=0.8，代表性初值 × 2 参数组
ALPHA = 0.8
for betaU, tag in [(0.6, 'endemic'), (0.1, 'free')]:
    frac_finals = []
    print(f"=== fractional alpha={ALPHA} beta^U={betaU} ===")
    for idx, (i_, a_) in enumerate(IC_LIST[::3]):  # 4 个代表性初值
        y0 = make_ic(i_, a_, 1000+idx)
        params = dict(N=N, mu=MU, delta=DELTA, lambd=LAM, beta_U=betaU, gamma=GAMMA,
                      alpha=ALPHA, m_consciousness=m_consciousness, m_risk=m_risk)
        from fast_caputo_solver import BlockConfig
        def make_F(A, B, params):
            N = params["N"]
            beta_U, beta_A = params["beta_U"], params["beta_U"]*params["gamma"]
            lambd, delta, mu = params["lambd"], params["delta"], params["mu"]
            def F(P, t):
                M = len(P)-1
                Pd = P[:-1,:].reshape(M, N, 3)
                BP, AP = np.zeros_like(Pd), np.zeros_like(Pd)
                for j in range(M):
                    BP[j] = A @ Pd[j]; AP[j] = B @ Pd[j]
                M3 = np.array([[0,0,0],[delta,-delta,0],[0,mu,-mu]])
                Q = np.zeros((N, M, 3, 3))
                for i in range(N):
                    e = np.zeros(N); e[i] = 1
                    for j in range(M):
                        s1 = e @ BP[j] @ np.array([[0],[1],[1]])
                        s2 = e @ AP[j] @ np.array([[0],[0],[1]])
                        M1 = s1 * np.array([[-lambd,lambd,0],[0,0,0],[0,0,0]])
                        M2 = s2 * np.array([[-beta_U,0,beta_U],[0,-beta_A,beta_A],[0,0,0]])
                        Q[i,j] = M3 + M1 + M2
                p = Pd.transpose(1,0,2)
                res = np.zeros((M, 3*N))
                for j in range(M):
                    res[j] = np.concatenate([p[i,j] @ Q[i,j] for i in range(N)])
                return res
            return F
        F = make_F(A, B, params)
        cfg = BlockConfig(16, 4, 2048, tol=1e-4, picard_omega=0.7, omega_backtrack=0.8)
        t0 = time.time()
        tf, yf = solve_caputo_blocked(F, ALPHA, y0, T, 2048, cfg)
        rho_fin = yf.reshape(-1, N, 3)[:, :, 2].mean()
        frac_finals.append(rho_fin)
        print(f"  IC({i_:.2f},{a_:.2f}) -> rho^AI = {rho_fin:.5f} ({time.time()-t0:.0f}s)")
    fa = np.array(frac_finals)
    print(f"  frac spread: range = {fa.max()-fa.min():.6f}")

# ---------- 绘图 ----------
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

colors = plt.cm.viridis(np.linspace(0, 1, len(IC_LIST)))
for tag, ax, title in [('endemic', axes[0], r'$\mathcal{R}_0>1$: convergence to the rumor-endemic equilibrium'),
                       ('free', axes[1], r'$\mathcal{R}_0<1$: convergence to the rumor-free equilibrium')]:
    for idx, (i_, a_, t, rho) in enumerate(results[tag]['trajs']):
        ax.plot(t, rho, color=colors[idx], lw=1.5, alpha=0.85)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\rho^{AI}$')
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.01, 0.6 if tag=='endemic' else 0.05)
    if tag == 'endemic':
        ax.axhline(y=results['endemic']['finals'][0], color='gray', ls='--', lw=1)
        ax.text(0.5, results['endemic']['finals'][0]+0.012, r'$\rho^{AI**}$', color='gray')

plt.tight_layout()
plt.savefig(_os.path.join(_DATA, 'global_stability', 'global_stability_verification.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(_os.path.join(_DATA, 'global_stability', 'global_stability_verification.png'), dpi=300, bbox_inches='tight')
print("\nFigure saved: global_stability_verification.pdf")

# 保存数值摘要
summary = {}
for tag in ['endemic', 'free']:
    finals = results[tag]['finals']
    summary[tag] = {'mean': float(np.mean(finals)), 'min': float(min(finals)),
                    'max': float(max(finals)), 'range': float(max(finals)-min(finals))}
with open(_os.path.join(_DATA, 'global_stability', 'global_stability_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary saved:", summary)
