# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# test_solver_ml.py
"""
Test fast_caputo_solver.solve_caputo_blocked against the analytic solution
y(t) = E_alpha(-lambda * t^alpha) for D^alpha y = -lambda y, y(0)=1.

Usage:
    python test_solver_ml.py
Requires:
    - numpy, matplotlib
    - prefer: mpmath (for reliable Mittag-Leffler). If not present, fallback is used.
"""
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# Import the solver you already have (ensure it's in the same dir or in PYTHONPATH)
try:
    from fast_caputo_solver import solve_caputo_blocked, BlockConfig
except Exception as e:
    raise RuntimeError("无法导入 fast_caputo_solver.py，请确认脚本与本文件在同一目录且文件名正确。原始错误: " + str(e))

# Try to import mpmath for robust Mittag-Leffler
_have_mpmath = False
try:
    import mpmath as mp
except Exception:
    _have_mpmath = False
    mp = None

# -------------------------
# Robust Mittag-Leffler helper
# -------------------------
def mittag_leffler_mp(alpha: float, z: complex) -> complex:
    """Use mpmath built-in (two-argument E_{alpha,1}(z) with beta=1)."""
    # mpmath has a function mp.mittag_leffler(z, alpha, beta) signature in many versions
    # but to be safe we try common variants.
    try:
        # mp.mittag_leffler(z, a) or mp.mittag_leffler(a, z)? Common is mp.mittag_leffler(z, a)
        return mp.mittag_leffler(z, alpha)
    except TypeError:
        try:
            # some versions: mp.mittag_leffler(alpha, z)
            return mp.mittag_leffler(alpha, z)
        except Exception as e:
            # fallback to general two-parameter with beta=1
            return mp.mittag_leffler(z, alpha, 1)

def mittag_leffler_series(alpha: float, z: complex, tol: float = 1e-16, max_terms: int = 2000) -> complex:
    """
    Plain power-series: E_alpha(z) = sum_{k=0}^\infty z^k / Gamma(alpha*k + 1).
    Good when |z| is modest (|z| <= ~1). Adaptively stops when terms small.
    """
    z = complex(z)
    s = 0+0j
    term = 1+0j
    s += term
    k = 1
    while k < max_terms:
        # compute next term carefully
        # term_k = z^k / Gamma(alpha*k + 1)
        # to avoid recomputing gamma at each step, compute term incrementally:
        # term_k = term_{k-1} * z * Gamma(alpha*(k-1)+1) / Gamma(alpha*k + 1)
        # but gamma ratio is still needed; for clarity we compute gamma each time
        denom = math.gamma(alpha * k + 1.0)
        term = (z ** k) / denom
        s += term
        if abs(term) < tol * max(1.0, abs(s)):
            break
        k += 1
    else:
        # didn't converge within max_terms
        # Return partial sum but warn
        print(f"WARNING: series for E_{alpha}({z}) didn't converge after {max_terms} terms (|z|={abs(z):.3g}).", file=sys.stderr)
    return s

def mittag_leffler(alpha: float, z: complex) -> complex:
    """
    Wrapper: prefer mpmath; else fallback to series with domain restriction.
    For large |z| we warn user about possible inaccuracy.
    """
    if _have_mpmath:
        # use mpmath with increased precision for safety
        prev_dps = mp.mp.dps
        try:
            mp.mp.dps = 50  # 50 decimal digits should be plenty for most tests
            val = mittag_leffler_mp(alpha, z)
            return complex(val)
        finally:
            mp.mp.dps = prev_dps
    else:
        # fallback
        if abs(z) > 2.0:
            print("WARNING: computing Mittag-Leffler by plain series with |z|>2 may be inaccurate. "
                  "Install mpmath for robust results.", file=sys.stderr)
        return mittag_leffler_series(alpha, z, tol=1e-14, max_terms=500)

# -------------------------
# Test problem parameters
# -------------------------
alpha = 0.6        # fractional order
lam = 1.0          # lambda > 0 in D^alpha y = -lambda y
y0 = 1.0
T = 1.0
N = 2000           # grid steps; increase for finer resolution

# Define F: solver expects F(y_block, t_block) where y_block shape (n_points, d)
def F_block(y_block, t_block):
    # scalar test; y_block is shape (m+1,) or (m+1,1) depending on implementation
    # return shape (m, ) or (m,1)
    # we use -lam * y at grid points k=0..m-1
    if y_block.ndim == 1:
        return -lam * y_block[:-1]
    else:
        # (m+1,d) -> return (m,d)
        return -lam * y_block[:-1, :]

# Solver config: keep reasonably strict tolerance
cfg = BlockConfig(block_steps=512, min_block_steps=64, max_block_steps=4096,
                  grow_factor=1.6, shrink_factor=0.5, target_iter=6,
                  max_iter=80, tol=1e-11, verbose=False)

# Solve numerically
t_grid, y_grid = solve_caputo_blocked(F_block, alpha, y0, T, N, cfg)
# y_grid shape: (N+1, d) ; for scalar problem d=1 or d=?
# Ensure we extract scalar series
if y_grid.ndim == 2 and y_grid.shape[1] == 1:
    y_num = y_grid[:,0]
elif y_grid.ndim == 2 and y_grid.shape[1] > 1:
    # if solver returned vector, and we used scalar initial y0, take first component
    y_num = y_grid[:,0]
else:
    y_num = np.asarray(y_grid).reshape(-1)

# Compute reference (analytic) solution
y_exact = np.empty_like(t_grid, dtype=float)
for i, tt in enumerate(t_grid):
    z = -lam * (tt ** alpha)
    val = mittag_leffler(alpha, z)
    # numerically sometimes gets slight complex residual; take real part if imag tiny
    if abs(val.imag) < 1e-12:
        val = val.real
    else:
        # if large imaginary part appears, warn
        print(f"WARNING: Mittag-Leffler returned complex value at t={tt}: {val}", file=sys.stderr)
        val = val.real
    y_exact[i] = float(val)

# Error
err = np.abs(y_num - y_exact)
max_err = np.max(err)
rms_err = np.sqrt(np.mean(err**2))

print(f"alpha={alpha}, lambda={lam}, T={T}, N={N}")
print(f"max abs error = {max_err:.3e}, RMS error = {rms_err:.3e}")

# Plot numeric vs exact and error
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t_grid, y_num, label='numeric', lw=1)
plt.plot(t_grid, y_exact, '--', label='exact (Mittag-Leffler)', lw=1)
plt.xlabel('t'); plt.ylabel('y(t)')
plt.title('Solution: numeric vs exact')
plt.legend()

plt.subplot(1,2,2)
plt.plot(t_grid, err)
plt.yscale('log')
plt.xlabel('t'); plt.ylabel('abs error')
plt.title('Absolute error (log scale)')

plt.tight_layout()
plt.savefig(_os.path.join(_FIGD, "tests", "test_solver_vs_mittag_leffler.png"), dpi=200)
print("figure saved to figures/tests/")
