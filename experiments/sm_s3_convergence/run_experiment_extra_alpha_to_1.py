# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# check_convergence_to_integer_with_inset.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from integer_order_solver import UAU_SIS_IntegerSolver, NetworkType

# ---------------------------
# 配置参数 (必须与分数阶仿真时完全一致)
# ---------------------------
N = 1000
m_consciousness = 4
m_risk = 2
T = 20.0
N_steps = 2048 * 2

params = {
    'lambd': 0.3,
    'beta_U': 0.6,
    'gamma': 1/3,
    'delta': 0.5,
    'mu': 1.0
}

fractional_root = _os.path.join(_DATA, "results_fractional")
output_fig = _os.path.join(_FIGD, "convergence_alpha_to_1_with_error.pdf")
alphas_frac = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# ---------------------------
# 1. 读取分数阶结果
# ---------------------------
def load_fractional_infection(alpha, beta_U, root_dir):
    case_dir = os.path.join(root_dir, f"alpha_{alpha}_betaU_{beta_U}")
    t_path = os.path.join(case_dir, "t.npy")
    y_path = os.path.join(case_dir, "y.npy")
    if not os.path.exists(t_path) or not os.path.exists(y_path):
        print(f"Warning: missing data for alpha={alpha}, betaU={beta_U}")
        return None, None
    t = np.load(t_path)
    y = np.load(y_path)
    I_avg = y.reshape(-1, N, 3)[:, :, 2].mean(axis=1)
    return t, I_avg

fractional_data = {}
for alpha in alphas_frac:
    t, I = load_fractional_infection(alpha, params['beta_U'], fractional_root)
    if t is not None:
        fractional_data[alpha] = (t, I)
        print(f"Loaded alpha={alpha}, final I={I[-1]:.4f}")

if len(fractional_data) == 0:
    raise RuntimeError("No fractional data found. Please run main.py first.")

# ---------------------------
# 2. 计算整数阶 (α=1) 结果
# ---------------------------
print("\nRunning integer-order (α=1) simulation...")
solver = UAU_SIS_IntegerSolver(
    N=N,
    network_type_consciousness=NetworkType.BA,
    network_type_risk=NetworkType.BA,
    network_params_consciousness={'m': m_consciousness},
    network_params_risk={'m': m_risk},
    seed=42
)
solver.set_parameters(**params)

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

y0_int = generate_initial_conditions(N, dominant_state=0, dominance=0.8, seed=42)

t_eval = np.linspace(0, T, N_steps+1)
result = solver.solve(
    t_span=(0, T),
    t_eval=t_eval,
    initial_conditions=y0_int,
    show_progress=True
)
t_int = result['t']
I_int = result['rho_AI']
print(f"Integer-order done, final I={I_int[-1]:.4f}")

# ---------------------------
# 3. 计算误差 (L2范数)
# ---------------------------
I_int_func = interp1d(t_int, I_int, kind='linear', fill_value='extrapolate')
errors = []
alphas_sorted = sorted(fractional_data.keys())
for alpha in alphas_sorted:
    t, I_frac = fractional_data[alpha]
    I_int_interp = I_int_func(t)
    mask = t > 0.1
    err = np.sqrt(np.mean((I_frac[mask] - I_int_interp[mask])**2))
    errors.append(err)
    print(f"α={alpha}: RMSE (t>0.1) = {err:.5f}")

# ... 前面的部分完全相同 ...

# 绘制主图
plt.rcParams.update({'font.size': 12})
fig, ax_main = plt.subplots(figsize=(10, 6))

# 分数阶曲线
colors = plt.cm.viridis(np.linspace(0, 1, len(fractional_data)))
for idx, (alpha, (t, I)) in enumerate(sorted(fractional_data.items())):
    mask = t > 0.01
    ax_main.semilogx(t[mask], I[mask], color=colors[idx], lw=2,
                     label=f'α={alpha}')

# 整数阶曲线
mask_int = t_int > 0.01
ax_main.semilogx(t_int[mask_int], I_int[mask_int], color='red', lw=3, linestyle='--',
                 label='α=1')

ax_main.set_xlabel('Time', fontsize=20)
ax_main.set_ylabel('AI state density', fontsize=20)
ax_main.legend(loc='best')
ax_main.grid(True, alpha=0.3)

# --- 嵌入子图：右下角 ---
# 位置 [left, bottom, width, height] (相对坐标)
inset_ax = fig.add_axes([0.62, 0.25, 0.3, 0.3])   # 右下角
inset_ax.plot(alphas_sorted, errors, 'o-', color='darkblue', lw=2, markersize=6)
inset_ax.set_xlabel('$\\alpha$', fontsize=20)
inset_ax.set_ylabel('RMSE', fontsize=20)   # 明确注明包含瞬态
inset_ax.set_title('Convergence error', fontsize=20)
inset_ax.grid(True, alpha=0.3, linestyle='--')
# 可选: inset_ax.set_yscale('log') 如果误差跨度大

plt.tight_layout()
plt.savefig(output_fig, dpi=300)
plt.show()
print(f"Figure saved to {output_fig}")