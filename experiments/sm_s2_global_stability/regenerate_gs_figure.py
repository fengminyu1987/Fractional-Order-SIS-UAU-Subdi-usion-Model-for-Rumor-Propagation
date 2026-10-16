# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# -*- coding: utf-8 -*-
"""Regenerate the global-stability verification figure as two separate panels.

Panel 1: endemic regime (beta^U = 0.6, R0 = 2.07 > 1).
Panel 2: rumor-free regime (beta^U = 0.1, R0 = 0.35 < 1).

Outputs:
    global_stability_endemic.pdf / .png
    global_stability_free.pdf    / .png

The in-figure titles are intentionally omitted; titles are provided by the
\\subfigure captions in the Supplementary Material. Font sizes are set to at
least the body-text size (10 pt) so that the panels are readable at column
width.
"""
import os
import time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fast_caputo_solver import solve_caputo_blocked, BlockConfig
from integer_order_solver import UAU_SIS_IntegerSolver, NetworkType

N = 250
m_consciousness, m_risk = 4, 2
SEED = 42
T = 20.0
LAM, DELTA, MU, GAMMA = 0.3, 0.5, 1.0, 1 / 3
ALPHA = 0.8

np.random.seed(SEED)
Gc = nx.barabasi_albert_graph(N, m_consciousness, seed=SEED)
Gr = nx.barabasi_albert_graph(N, m_risk, seed=SEED + 1)
A = nx.to_numpy_array(Gc)
B = nx.to_numpy_array(Gr)


def make_ic(inf_frac, awr_frac, seed):
    rng = np.random.default_rng(seed)
    y0 = np.zeros(3 * N)
    for i in range(N):
        p = np.array([1 - inf_frac - awr_frac, awr_frac, inf_frac])
        p += rng.normal(0, 0.02, 3)
        p = np.clip(p, 0, 1)
        p = p / p.sum()
        y0[3 * i:3 * i + 3] = p
    return y0


def make_F(beta_U):
    beta_A = beta_U * GAMMA

    def F(P, t):
        M = len(P) - 1
        Pd = P[:-1, :].reshape(M, N, 3)
        BP, AP = np.zeros_like(Pd), np.zeros_like(Pd)
        for j in range(M):
            BP[j] = A @ Pd[j]
            AP[j] = B @ Pd[j]
        M3 = np.array([[0, 0, 0], [DELTA, -DELTA, 0], [0, MU, -MU]])
        Q = np.zeros((N, M, 3, 3))
        for i in range(N):
            e = np.zeros(N)
            e[i] = 1
            for j in range(M):
                s1 = e @ BP[j] @ np.array([[0], [1], [1]])
                s2 = e @ AP[j] @ np.array([[0], [0], [1]])
                M1 = s1 * np.array([[-LAM, LAM, 0], [0, 0, 0], [0, 0, 0]])
                M2 = s2 * np.array([[-beta_U, 0, beta_U], [0, -beta_A, beta_A], [0, 0, 0]])
                Q[i, j] = M3 + M1 + M2
        p = Pd.transpose(1, 0, 2)
        res = np.zeros((M, 3 * N))
        for j in range(M):
            res[j] = np.concatenate([p[i, j] @ Q[i, j] for i in range(N)])
        return res
    return F


def run_fractional(beta_U, inf_frac, awr_frac, seed):
    y0 = make_ic(inf_frac, awr_frac, seed)
    F = make_F(beta_U)
    cfg = BlockConfig(16, 4, 2048, tol=1e-4, picard_omega=0.7, omega_backtrack=0.8)
    t0 = time.time()
    tf, yf = solve_caputo_blocked(F, ALPHA, y0, T, 2048, cfg)
    rho = yf.reshape(-1, N, 3)[:, :, 2].mean(axis=1)
    print(f"  frac betaU={beta_U} IC=({inf_frac},{awr_frac}) -> {rho[-1]:.6f} ({time.time()-t0:.0f}s)",
          flush=True)
    return tf, rho


def run_integer(beta_U, inf_frac, awr_frac, seed):
    solver = UAU_SIS_IntegerSolver(
        N=N,
        network_type_consciousness=NetworkType.BA,
        network_type_risk=NetworkType.BA,
        network_params_consciousness={"m": m_consciousness},
        network_params_risk={"m": m_risk},
        seed=SEED,
    )
    solver.set_custom_networks(A, B)
    solver.set_parameters(lambd=LAM, beta_U=beta_U, gamma=GAMMA, delta=DELTA, mu=MU)
    y0 = make_ic(inf_frac, awr_frac, seed)
    res = solver.solve(t_span=(0, T), t_eval=np.linspace(0, T, 401),
                       initial_conditions=y0, show_progress=False)
    return res["t"], res["rho_AI"]


IC_LIST = [
    (0.01, 0.01), (0.01, 0.30), (0.01, 0.60),
    (0.05, 0.01), (0.05, 0.30), (0.05, 0.60),
    (0.15, 0.01), (0.15, 0.30), (0.15, 0.60),
    (0.30, 0.01), (0.30, 0.30), (0.30, 0.60),
]
FRAC_ICS_END = [(0.01, 0.01), (0.05, 0.30), (0.15, 0.60), (0.30, 0.01)]
FRAC_ICS_FREE = [(0.05, 0.30), (0.30, 0.01)]

# ---------- fractional trajectories (use cache when available) ----------
print("Fractional trajectories (alpha = 0.8) ...", flush=True)
for idx, (i_, a_) in enumerate(FRAC_ICS_END):
    fn = _os.path.join(_DATA, 'global_stability', f"gs_frac_endemic_{idx}.npy")
    if not os.path.exists(fn):
        tf, rho = run_fractional(0.6, i_, a_, 2000 + idx)
        np.save(fn, np.vstack([tf, rho]))
    else:
        print(f"  endemic IC={i_, a_} cached", flush=True)
for idx, (i_, a_) in enumerate(FRAC_ICS_FREE):
    fn = _os.path.join(_DATA, 'global_stability', f"gs_frac_free_{idx}.npy")
    if not os.path.exists(fn):
        tf, rho = run_fractional(0.1, i_, a_, 3000 + idx)
        np.save(fn, np.vstack([tf, rho]))
    else:
        print(f"  free IC={i_, a_} cached", flush=True)

# ---------- integer trajectories ----------
print("Regenerating integer-order trajectories ...", flush=True)
int_end = [run_integer(0.6, i_, a_, 1000 + idx) for idx, (i_, a_) in enumerate(IC_LIST)]
int_free = [run_integer(0.1, i_, a_, 1000 + idx) for idx, (i_, a_) in enumerate(IC_LIST)]

# ---------- styling ----------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.1,
})
colors_int = plt.cm.viridis(np.linspace(0, 1, len(IC_LIST)))
colors_frac = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
legend_handles = [
    Line2D([0], [0], color="gray", lw=1.4, label="Integer-order (12 ICs)"),
    Line2D([0], [0], color="black", lw=1.5, ls="--",
           label="Fractional-order ($\\alpha=0.8$)"),
]


def style_ax(ax, ylim, ref_line=None, with_legend=False):
    ax.grid(True, alpha=0.3)
    ax.set_ylim(*ylim)
    if ref_line is not None:
        ax.axhline(y=ref_line, color="gray", ls=":", lw=1.2)
    if with_legend:
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95,
                  ncol=1, handlelength=1.4, borderpad=0.5,
                  labelspacing=0.35, fontsize=9)


# ---------- Panel 1: endemic ----------
fig1, ax1 = plt.subplots(figsize=(3.4, 2.6))
for idx, (i_, a_) in enumerate(IC_LIST):
    t, rho = int_end[idx]
    ax1.plot(t, rho, color=colors_int[idx], lw=0.9, alpha=0.8)
for idx in range(len(FRAC_ICS_END)):
    data = np.load(_os.path.join(_DATA, 'global_stability', f"gs_frac_endemic_{idx}.npy"))
    ax1.plot(data[0], data[1], color=colors_frac[idx], lw=1.5, ls="--", alpha=0.95)
style_ax(ax1, (-0.01, 0.55), ref_line=0.23008, with_legend=True)
ax1.set_xlabel("Time")
ax1.set_ylabel(r"$\rho^{AI}$")
plt.tight_layout()
fig1.savefig(_os.path.join(_FIGD, "global_stability_endemic.pdf"), dpi=300)
fig1.savefig(_os.path.join(_FIGD, "global_stability_endemic.png"), dpi=300)
plt.close(fig1)

# ---------- Panel 2: rumor-free ----------
fig2, ax2 = plt.subplots(figsize=(3.4, 2.6))
for idx, (i_, a_) in enumerate(IC_LIST):
    t, rho = int_free[idx]
    ax2.plot(t, rho, color=colors_int[idx], lw=0.9, alpha=0.8)
for idx in range(len(FRAC_ICS_FREE)):
    data = np.load(_os.path.join(_DATA, 'global_stability', f"gs_frac_free_{idx}.npy"))
    ax2.plot(data[0], data[1], color=colors_frac[idx], lw=1.5, ls="--", alpha=0.95)
style_ax(ax2, (-0.002, 0.05), with_legend=False)
ax2.set_xlabel("Time")
ax2.set_ylabel(r"$\rho^{AI}$")
plt.tight_layout()
fig2.savefig(_os.path.join(_FIGD, "global_stability_free.pdf"), dpi=300)
fig2.savefig(_os.path.join(_FIGD, "global_stability_free.png"), dpi=300)
plt.close(fig2)

print("Saved: global_stability_endemic.pdf / global_stability_free.pdf", flush=True)
