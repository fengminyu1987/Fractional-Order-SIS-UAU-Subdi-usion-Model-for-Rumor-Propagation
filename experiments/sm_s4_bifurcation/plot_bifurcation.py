# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# plot_bifurcation.py  (final)
# 分岔图：整数阶 ODE + 同网络高精度分数阶验证 → 图中图
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# ============================================
# 1. 整数阶分岔数据 (N=250, α=1, 40 β^U)
# ============================================
with open(_os.path.join(_DATA, "results_bifurcation", "integer_results.json")) as f:
    integer_results = json.load(f)
bU_all = sorted([float(k) for k in integer_results.keys()])
rho_all = [integer_results[str(b)]['rho_AI_final'] for b in bU_all]

# ============================================
# 2. 分数阶验证数据 (N=250, 同网络, 新跑)
# ============================================
frac_dir = _os.path.join(_DATA, "results_bifurcation_frac")
alphas_frac = [0.6, 0.8]

frac_points = {}
for alpha in alphas_frac:
    pts = []
    for entry in os.listdir(frac_dir):
        if not entry.startswith(f"alpha_{alpha}_betaU_"):
            continue
        case = os.path.join(frac_dir, entry)
        y_path = os.path.join(case, "y.npy")
        if not os.path.exists(y_path):
            continue
        y = np.load(y_path)
        N_fr = y.shape[1] // 3
        rho = y.reshape(-1, N_fr, 3)[:, :, 2].mean()
        # 从目录名提取 beta_U
        beta_U = float(entry.split("_betaU_")[1])
        pts.append((beta_U, rho))
    pts.sort(key=lambda x: x[0])
    if pts:
        frac_points[alpha] = pts

if not frac_points:
    print("Warning: No fractional data found in", frac_dir)

# ============================================
# 3. 绘制
# ============================================
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(figsize=(9, 6))

# ---------- 主图 ----------
ax.plot(bU_all, rho_all, 'o-', color='red', lw=2, ms=4, label=r'$\alpha=1$')

colors = plt.cm.viridis(np.linspace(0, 1, len(alphas_frac)))
for idx, alpha in enumerate(alphas_frac):
    if alpha in frac_points:
        bx, ry = zip(*frac_points[alpha])
        ax.scatter(bx, ry, s=50, color=colors[idx], zorder=5,
                   edgecolors='k', linewidths=0.5,
                   label=rf'$\alpha={alpha}$')

ax.set_xlabel(r'$\beta^U$', fontsize=20)
ax.set_ylabel(r'$\rho^{AI**}$', fontsize=20)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 2.1)
ax.set_ylim(-0.02, 0.7)

thresh_bU = 0.289
ax.axvline(x=thresh_bU, color='gray', lw=1.5, ls='--', dashes=(5, 5))
ax.text(thresh_bU + 0.02, 0.65, r'$\mathcal{R}_0 \approx 1$',
        fontsize=14, color='gray', fontstyle='italic', va='top')

# ---------- 图中图 ----------
x1, x2 = 0.0, 0.5
y1, y2 = -0.01, 0.12

inset = ax.inset_axes([0.55, 0.24, 0.378, 0.336])

mask_zoom = [(b >= x1) and (b <= x2) for b in bU_all]
bUz = [bU_all[i] for i in range(len(bU_all)) if mask_zoom[i]]
rhoz = [rho_all[i] for i in range(len(bU_all)) if mask_zoom[i]]
inset.plot(bUz, rhoz, 'o-', color='red', lw=2, ms=6)

for idx, alpha in enumerate(alphas_frac):
    if alpha in frac_points:
        pts_sub = [(b, r) for (b, r) in frac_points[alpha] if x1 <= b <= x2]
        if pts_sub:
            bx, ry = zip(*pts_sub)
            inset.scatter(bx, ry, s=45, color=colors[idx],
                          edgecolors='k', linewidths=0.5, zorder=5)

inset.set_xlim(x1, x2)
inset.set_ylim(y1, y2)
inset.set_ylabel(r'$\rho^{AI**}$', fontsize=12)
inset.grid(True, alpha=0.3)
inset.tick_params(labelsize=8)

inset.axvline(x=thresh_bU, color='gray', lw=1.2, ls='--', dashes=(5, 5))
inset.text(thresh_bU + 0.005, y2 * 0.88, r'$\mathcal{R}_0\approx1$',
           fontsize=9, color='gray', fontstyle='italic')

# 矩形框 + 连线
rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                      fill=False, edgecolor='gray', lw=1, ls='-')
ax.add_patch(rect)

con1 = ConnectionPatch(xyA=(x2, y1), coordsA='data', xyB=(x2, y1), coordsB='data',
                        axesA=ax, axesB=inset,
                        arrowstyle='-', color='gray', lw=1.2, shrinkB=3)
con2 = ConnectionPatch(xyA=(x1, y2), coordsA='data', xyB=(x1, y2), coordsB='data',
                        axesA=ax, axesB=inset,
                        arrowstyle='-', color='gray', lw=1.2, shrinkB=3)
ax.add_artist(con1)
ax.add_artist(con2)

plt.tight_layout()
out_dir = _FIGD
out_pdf = os.path.join(out_dir, "bifurcation_diagram_v3_new.pdf")
out_png = os.path.join(out_dir, "bifurcation_diagram_v3_new.png")
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved to {out_pdf}")

# 打印比较
print("\nComparison (same network, N=250):")
print(f"{'betaU':>8s}  {'alpha=1':>10s}  {'alpha=0.6':>10s}  {'alpha=0.8':>10s}")
print("-" * 50)
for bU in sorted(set([b for a in frac_points for b, _ in frac_points[a]])):
    v_int = integer_results.get(str(bU), {}).get('rho_AI_final', float('nan'))
    v06 = dict(frac_points.get(0.6, [])).get(bU, float('nan'))
    v08 = dict(frac_points.get(0.8, [])).get(bU, float('nan'))
    print(f"{bU:8.3f}  {v_int:10.6f}  {v06:10.6f}  {v08:10.6f}")
