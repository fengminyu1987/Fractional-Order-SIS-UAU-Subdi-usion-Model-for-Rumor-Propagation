# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_bifurcation.py
# 分岔图实验：稳态感染密度 ρ^AI 随 β^U 的变化（不同 α 的对比曲线）
import os
import time
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from integer_order_solver import UAU_SIS_IntegerSolver, NetworkType
from fast_caputo_solver import solve_caputo_blocked, BlockConfig

# ============================================
# 0. 配置
# ============================================
N = 50
m_consciousness = 4
m_risk = 2
SEED = 42

T = 20.0
N_steps = 2048

FIXED_PARAMS = dict(
    lambd=0.3,
    gamma=1/3,
    delta=0.5,
    mu=1.0,
)

BETA_U_SCAN = np.linspace(0.05, 2.0, 40)   # 40 个 β^U 点（整数阶全扫）
BETA_U_FRAC = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]  # 分数阶仅取关键点确认
ALPHAS_FRAC = [0.6, 0.8]                    # 分数阶 α（确认不影响稳态）
OUTPUT_DIR = _os.path.join(_DATA, "results_bifurcation")
OUTPUT_FIG = _os.path.join(_DATA, "results_bifurcation", "bifurcation_diagram.pdf")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# 1. 生成固定网络
# ============================================
np.random.seed(SEED)
Gc = nx.barabasi_albert_graph(N, m_consciousness, seed=SEED)
Gr = nx.barabasi_albert_graph(N, m_risk, seed=SEED+1)
A = nx.to_numpy_array(Gc)
B = nx.to_numpy_array(Gr)

# ============================================
# 2. 初始条件
# ============================================
def generate_initial_conditions(N, dominant_state=0, dominance=0.8, seed=42):
    np.random.seed(seed)
    y0 = np.zeros(3 * N)
    for i in range(N):
        p = np.zeros(3)
        p[dominant_state] = dominance
        remaining = 1.0 - dominance
        r1, r2 = np.random.rand(2)
        s = r1 + r2
        other = [j for j in range(3) if j != dominant_state]
        p[other] = remaining * np.array([r1, r2]) / s
        y0[3*i:3*i+3] = p
    return y0

y0 = generate_initial_conditions(N, dominant_state=0, dominance=0.8, seed=SEED)

# ============================================
# 3. 整数阶扫描 (α=1) — 快速
# ============================================
print("=" * 60)
print("Phase 1: Integer-order (α=1) scan")
print("=" * 60)

integer_results = {}  # {beta_U: {rho_AI_final, ...}}

for i, beta_U in enumerate(BETA_U_SCAN):
    label = f"β^U={beta_U:.3f}"
    print(f"[{i+1}/{len(BETA_U_SCAN)}] Integer-order {label} ...", end=" ")

    params_run = {**FIXED_PARAMS, 'beta_U': beta_U}
    solver = UAU_SIS_IntegerSolver(
        N=N,
        network_type_consciousness=NetworkType.BA,
        network_type_risk=NetworkType.BA,
        network_params_consciousness={'m': m_consciousness},
        network_params_risk={'m': m_risk},
        seed=SEED
    )
    solver.set_custom_networks(A, B)
    solver.set_parameters(**params_run)

    t_eval = np.linspace(0, T, N_steps + 1)
    result = solver.solve(
        t_span=(0, T),
        t_eval=t_eval,
        initial_conditions=y0.copy(),
        show_progress=False
    )

    rho_AI_final = result['rho_AI'][-1]
    integer_results[float(beta_U)] = {
        'rho_AI_final': float(rho_AI_final),
        'rho_US_final': float(result['rho_US'][-1]),
        'rho_AS_final': float(result['rho_AS'][-1]),
        'beta_U': float(beta_U),
        'alpha': 1.0,
    }
    print(f"ρ^AI={rho_AI_final:.4f}")

# 保存整数阶结果
with open(os.path.join(OUTPUT_DIR, "integer_results.json"), "w") as f:
    json.dump(integer_results, f, indent=2)

# ============================================
# 4. 分数阶扫描 — 确认 α 不影响稳态
# ============================================
print("\n" + "=" * 60)
print("Phase 2: Fractional-order scan (α=0.6, 0.8)")
print("=" * 60)

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
            e = np.zeros(N); e[i] = 1
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

fractional_results = {}

for alpha in ALPHAS_FRAC:
    fractional_results[alpha] = {}
    for i, beta_U in enumerate(BETA_U_FRAC):
        label = f"α={alpha}, β^U={beta_U:.3f}"
        case_dir = os.path.join(OUTPUT_DIR, f"alpha_{alpha}_betaU_{beta_U:.3f}")
        if os.path.exists(os.path.join(case_dir, "y.npy")):
            print(f"[{i+1}/{len(BETA_U_FRAC)}] {label} (cached)")
            y = np.load(os.path.join(case_dir, "y.npy"))
            rho_AI = y.reshape(-1, N, 3)[:, :, 2].mean()
        else:
            print(f"[{i+1}/{len(BETA_U_FRAC)}] {label} ...", end=" ", flush=True)
            params_run = {**FIXED_PARAMS, 'beta_U': beta_U, 'N': N,
                          'm_consciousness': m_consciousness, 'm_risk': m_risk}
            F = make_F(A, B, params_run)
            cfg = BlockConfig(block_steps=256, tol=1e-4, verbose=False,
                              picard_omega=0.7, omega_backtrack=0.8)
            t0 = time.time()
            t_arr, y_arr = solve_caputo_blocked(F, alpha, y0.copy(), T, N_steps, cfg)
            elapsed = time.time() - t0
            rho_AI = y_arr.reshape(-1, N, 3)[:, :, 2].mean()
            os.makedirs(case_dir, exist_ok=True)
            np.save(os.path.join(case_dir, "t.npy"), t_arr)
            np.save(os.path.join(case_dir, "y.npy"), y_arr)
            with open(os.path.join(case_dir, "params.json"), "w") as f:
                json.dump({**params_run, 'alpha': alpha}, f, indent=2)
            print(f"ρ^AI={rho_AI:.4f} ({elapsed:.0f}s)")

        fractional_results[alpha][float(beta_U)] = {
            'rho_AI_final': float(rho_AI),
            'beta_U': float(beta_U),
            'alpha': float(alpha),
        }

# 保存分数阶结果
with open(os.path.join(OUTPUT_DIR, "fractional_results.json"), "w") as f:
    json.dump(fractional_results, f, indent=2)

# ============================================
# 5. 绘制分岔图
# ============================================
print("\n" + "=" * 60)
print("Phase 3: Plotting bifurcation diagram")
print("=" * 60)

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(figsize=(8, 5.5))

# 整数阶曲线 (α=1)
bU_vals = sorted(integer_results.keys())
rho_vals = [integer_results[b]['rho_AI_final'] for b in bU_vals]
ax.plot(bU_vals, rho_vals, 'o-', color='red', lw=2, ms=5, label=r'$\alpha=1$ (integer-order)')

# 分数阶曲线
colors = ['#1f77b4', '#2ca02c']
styles = ['s--', '^--']
for idx, alpha in enumerate(ALPHAS_FRAC):
    if alpha in fractional_results:
        bU_f = sorted(fractional_results[alpha].keys())
        rho_f = [fractional_results[alpha][b]['rho_AI_final'] for b in bU_f]
        ax.plot(bU_f, rho_f, styles[idx], color=colors[idx], lw=1.5, ms=5,
                label=rf'$\alpha={alpha}$ (fractional)')

# 标记 R0=1 阈值线（可后续从理论算出精确值后替换）
ax.axhline(y=0, color='gray', lw=0.5, ls='-')

ax.set_xlabel(r'$\beta^U$ (Infection rate for unaware)', fontsize=16)
ax.set_ylabel(r'Steady-state $\rho^{AI}$', fontsize=16)
ax.set_title('Bifurcation diagram of the Fractional-Order UAU--SIS Model', fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_FIG.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"\nFigure saved to {OUTPUT_FIG}")
plt.show()

print("\nDone. All results saved to", OUTPUT_DIR)
