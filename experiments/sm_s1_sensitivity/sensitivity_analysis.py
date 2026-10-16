# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# sensitivity_analysis.py
"""
Parameter sensitivity analysis — tornado plot using existing simulation data.
Reads pre-computed results from:
  - results_fractional/          (fractional-order alpha x betaU, N=1000)
  - int_UAU_SIS/results_experiment3/  (lambda x delta x m,  N=250)

No new simulations required.  Output: sensitivity_tornado.pdf
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob as _glob

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ── locate project root ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = SCRIPT_DIR

FRAC_DIR = _os.path.join(_DATA, "results_fractional")
EXP3_DIR = _os.path.join(_DATA, "results_experiment3")


def _load_final_AI_npy(folder, N):
    yp = os.path.join(folder, "y.npy")
    if not os.path.exists(yp):
        return None
    y = np.load(yp)
    return float(np.mean(y[-1].reshape(N, 3)[:, 2]))


def _load_final_AI_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath) as fh:
        d = json.load(fh)
    v = d.get("final_AI_density", -1)
    return v if v >= 0 else None


# ── gather alpha & beta_U from fractional data ────────────────────────
def gather_alpha():
    out = []
    for a in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        rho = _load_final_AI_npy(
            os.path.join(FRAC_DIR, f"alpha_{a}_betaU_0.6"), 1000)
        if rho is not None:
            out.append((a, rho))
    return out


def gather_betaU():
    out = []
    for b in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        rho = _load_final_AI_npy(
            os.path.join(FRAC_DIR, f"alpha_0.8_betaU_{b}"), 1000)
        if rho is not None:
            out.append((b, rho))
    return out


# ── gather lambda / delta / m  from integer experiment3 data ──────────
def gather_exp3(param, **fixed):
    """
    param : 'lambd', 'delta', or 'm_consciousness'
    fixed : dict fixing the other two parameters, e.g. delta=0.5, m_consciousness=4
    """
    scan_range = {
        'lambd': np.round(np.linspace(0.1, 1.0, 10), 2),
        'delta': np.round(np.linspace(0.1, 1.0, 10), 2),
        'm_consciousness': [2, 4, 8],
    }[param]

    out = []
    for v in scan_range:
        vals = fixed.copy()
        vals[param] = v
        m = vals['m_consciousness']
        lam = vals.get('lambd', 0.5)
        d = vals.get('delta', 0.5)
        fname = f"m{m}_lambd{lam}_delta{d}.json"
        fpath = os.path.join(EXP3_DIR, f"m{m}", fname)
        rho = _load_final_AI_json(fpath)
        if rho is not None:
            out.append((v, rho))
    return out


# ── build parameter dictionary ────────────────────────────────────────
def build_ranges():
    effects = {}

    # alpha (fractional)
    row = gather_alpha()
    if len(row) >= 2:
        vals = [r[1] for r in row]
        baseline_idx = [i for i, r in enumerate(row) if r[0] == 0.8][0]
        effects[r"$\alpha$"] = (min(vals), max(vals), vals[baseline_idx])

    # beta_U (fractional)
    row = gather_betaU()
    if len(row) >= 2:
        vals = [r[1] for r in row]
        baseline_idx = [i for i, r in enumerate(row) if r[0] == 0.6][0]
        effects[r"$\beta^U$"] = (min(vals), max(vals), vals[baseline_idx])

    # lambda (integer, fix delta=0.5, m=4)
    row = gather_exp3('lambd', delta=0.5, m_consciousness=4)
    if len(row) >= 2:
        vals = [r[1] for r in row]
        bl = [r[1] for r in row if abs(r[0] - 0.5) < 1e-6]
        baseline = bl[0] if bl else np.median(vals)
        effects[r"$\lambda$"] = (min(vals), max(vals), baseline)

    # delta (integer, fix lambd=0.5, m=4)
    row = gather_exp3('delta', lambd=0.5, m_consciousness=4)
    if len(row) >= 2:
        vals = [r[1] for r in row]
        bl = [r[1] for r in row if abs(r[0] - 0.5) < 1e-6]
        baseline = bl[0] if bl else np.median(vals)
        effects[r"$\delta$"] = (min(vals), max(vals), baseline)

    # m_consciousness (integer, fix lambd=0.5, delta=0.5)
    row = gather_exp3('m_consciousness', lambd=0.5, delta=0.5)
    if len(row) >= 2:
        vals = [r[1] for r in row]
        bl = [r[1] for r in row if abs(r[0] - 4) < 1e-6]
        baseline = bl[0] if bl else np.median(vals)
        effects[r"$m_A$"] = (min(vals), max(vals), baseline)

    return effects


# ── tornado plot ──────────────────────────────────────────────────────
def plot_tornado(effects, outfile="sensitivity_tornado.pdf"):
    names = list(effects.keys())
    n = len(names)
    if n == 0:
        print("No data to plot.")
        return

    baselines = np.array([effects[k][2] for k in names])
    mins      = np.array([effects[k][0] for k in names])
    maxs      = np.array([effects[k][1] for k in names])
    spans     = maxs - mins

    # sort ascending by span
    order = np.argsort(spans)
    names    = [names[i] for i in order]
    baselines = baselines[order]
    mins      = mins[order]
    maxs      = maxs[order]
    spans     = spans[order]

    ypos = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bar_h = 0.55

    # Left whisker  (min → baseline)
    ax.barh(ypos, baselines - mins, height=bar_h, left=mins,
            color='#d62728', edgecolor='white', linewidth=0.3, alpha=0.85,
            label='Decrease from baseline')
    # Right whisker (baseline → max)
    ax.barh(ypos, maxs - baselines, height=bar_h, left=baselines,
            color='#1f77b4', edgecolor='white', linewidth=0.3, alpha=0.85,
            label='Increase from baseline')

    # ── baseline vertical line (data coordinates, aligned to each bar) ──
    for i in range(n):
        ax.plot([baselines[i], baselines[i]],
                [ypos[i] - bar_h/2, ypos[i] + bar_h/2],
                color='black', linewidth=1.5, zorder=4)
    # dummy entry for legend
    ax.plot([], [], color='black', linewidth=1.5, label='Baseline value')

    # ── text labels ──
    x_pad = (max(maxs) - min(mins)) * 0.025
    for i in range(n):
        # left-end value
        ax.text(mins[i] - x_pad, ypos[i], f'{mins[i]:.3f}',
                ha='right', va='center', fontsize=12, color='#555555')
        # right-end value
        ax.text(maxs[i] + x_pad, ypos[i], f'{maxs[i]:.3f}',
                ha='left',  va='center', fontsize=12, color='#555555')
        # span annotation
        ax.text(maxs[i] + x_pad * 4.5, ypos[i],
                f'Δ={spans[i]:.3f}',
                ha='left', va='center', fontsize=12, color='#333333',
                style='italic')

    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel(r'$\rho^{AI**}$')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='x', alpha=0.25)

    # Expand x-limits to fit all text
    ax.set_xlim(min(mins) - x_pad * 12,
                max(maxs) + x_pad * 18)

    plt.tight_layout()
    out_pdf = _os.path.join(_FIGD, "sensitivity_tornado.pdf")
    out_png = _os.path.join(_FIGD, "sensitivity_tornado.png")
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


# ── main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    effects = build_ranges()
    if not effects:
        print("ERROR: no data found.")
    else:
        print("\n=== Parameter sensitivity ranges ===")
        print(f"{'Parameter':<30s} {'Min':>8s} {'Max':>8s} {'Span':>8s} {'Baseline':>8s}")
        print("-" * 72)
        for name, (lo, hi, bl) in effects.items():
            print(f"{name:<30s} {lo:8.4f} {hi:8.4f} {hi-lo:8.4f} {bl:8.4f}")
        plot_tornado(effects, outfile="sensitivity_tornado.pdf")
