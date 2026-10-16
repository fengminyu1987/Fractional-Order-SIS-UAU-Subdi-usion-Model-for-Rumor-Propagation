# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# postprocess_exp2_fixed_us_as_ai.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Try LaTeX
try:
    plt.rcParams["text.usetex"] = True
except Exception:
    plt.rcParams["text.usetex"] = False
plt.rcParams["font.size"] = 14

RESULTS_DIR = _os.path.join(_DATA, "results_fractional")

def load_case(case_dir):
    t = np.load(os.path.join(case_dir, "t.npy"))
    y = np.load(os.path.join(case_dir, "y.npy"))
    with open(os.path.join(case_dir, "params.json")) as f:
        params = json.load(f)
    return t, y, params

def final_infected_fraction_us_as_ai(y):
    """
    Interpret y as shape (T, 3N) with [US, AS, AI].
    Final infected fraction = average of AI at final time.
    """
    AI = y[:, 2::3]
    return AI.mean(axis=1)[-1]

def series_US_AS_AI(y):
    """Return average time series of US, AS, AI."""
    US = y[:, 0::3].mean(axis=1)
    AS = y[:, 1::3].mean(axis=1)
    AI = y[:, 2::3].mean(axis=1)
    return US, AS, AI

def scan_alpha_beta_dirs(results_dir=RESULTS_DIR):
    grid = {}
    alphas, betas = set(), set()
    for name in os.listdir(results_dir):
        if not name.startswith("alpha_"):
            continue
        parts = name.split("_")
        if len(parts) != 4 or parts[2] != "betaU":
            continue
        try:
            alpha = float(parts[1])
            betaU = float(parts[3])
        except ValueError:
            continue
        case_dir = os.path.join(results_dir, name)
        try:
            t, y, _ = load_case(case_dir)
            val = final_infected_fraction_us_as_ai(y)
            grid[(alpha, betaU)] = val
            alphas.add(alpha)
            betas.add(betaU)
        except Exception as e:
            print(f"[WARN] skip {name}: {e}")
    alphas_sorted = sorted(list(alphas))
    betas_sorted  = sorted(list(betas))
    return grid, alphas_sorted, betas_sorted

def plot_heatmap(grid, alphas_sorted, betas_sorted, out_file=_os.path.join(_FIGD, "heatmap_exp2_us_as_ai.pdf")):
    Z = np.full((len(alphas_sorted), len(betas_sorted)), np.nan)
    for i,a in enumerate(alphas_sorted):
        for j,b in enumerate(betas_sorted):
            if (a,b) in grid:
                Z[i,j] = grid[(a,b)]
    Z_masked = np.ma.masked_invalid(Z)
    cmap = cm.get_cmap("viridis")
    cmap.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(Z_masked, origin="lower", cmap=cmap, aspect="auto",
                   extent=[betas_sorted[0], betas_sorted[-1], alphas_sorted[0], alphas_sorted[-1]],
                   interpolation="nearest")
    ax.set_xlabel(r"$\beta^{U}$", fontsize=24)
    ax.set_ylabel(r"Fractional order $\alpha$", fontsize=24)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Final infected (AI fraction)", fontsize=24)

    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] heatmap saved to {out_file}")

def plot_three_trajectories(cases, out_file=_os.path.join(_FIGD, "trajectories_US_AS_AI.png")):
    fig, axes = plt.subplots(1, len(cases), figsize=(5*len(cases),4), sharey=True)
    if len(cases) == 1: axes = [axes]

    for ax, (dirname, label) in zip(axes, cases):
        case_dir = os.path.join(RESULTS_DIR, dirname)
        t, y, _ = load_case(case_dir)
        US, AS, AI = series_US_AS_AI(y)

        ax.plot(t, US, label="US", lw=1.8)
        ax.plot(t, AS, label="AS", lw=1.8)
        ax.plot(t, AI, label="AI", lw=1.8)

        ax.set_xlabel(r"Time $t$")
        ax.set_title(label)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Average probability")
    axes[-1].legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] trajectories saved to {out_file}")

if __name__ == "__main__":
    grid, alphas_sorted, betas_sorted = scan_alpha_beta_dirs(RESULTS_DIR)
    plot_heatmap(grid, alphas_sorted, betas_sorted)

    # representative cases (make sure dirs exist)
    cases = [
        ("alpha_0.9_betaU_0.3", r"$\alpha=0.9,\ \beta^{U}=0.3$"),  # subcritical
        ("alpha_0.7_betaU_0.6", r"$\alpha=0.7,\ \beta^{U}=0.6$"),  # near-critical
        ("alpha_0.6_betaU_1.0", r"$\alpha=0.6,\ \beta^{U}=1.0$"),  # supercritical
    ]
    plot_three_trajectories(cases)
