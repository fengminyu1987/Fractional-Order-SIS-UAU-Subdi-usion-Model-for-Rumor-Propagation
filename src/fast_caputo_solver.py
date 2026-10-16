"""
Enhanced Fast solver for Caputo fractional ODEs
(vector-valued, adaptive blocked version with stabilized Picard iteration)
"""
from __future__ import annotations
import numpy as np
from numpy.fft import rfft, irfft
import math
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

Array = np.ndarray

# ========================
# Utility helpers
# ========================

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def _fft_convolve_columns(b: Array, Fmat: Array) -> Array:
    """Convolve 1D kernel b with each column of Fmat using FFT.
    b: (L_b,), Fmat: (L_f, d) -> returns full conv (L_b+L_f-1, d)
    """
    if Fmat.ndim == 1:
        Fmat = Fmat[:, None]
    Lb = len(b)
    Lf, d = Fmat.shape
    n = Lb + Lf - 1
    nfft = _next_pow2(n)
    B = rfft(b, nfft)
    C = np.empty((nfft//2 + 1, d), dtype=complex)
    for j in range(d):
        Fj = rfft(Fmat[:, j], nfft)
        C[:, j] = B * Fj
    c = np.empty((nfft, d))
    for j in range(d):
        c[:, j] = irfft(C[:, j], nfft)
    return c[:n, :]

def _ensure_2d_y0(y0: Array | float) -> Array:
    y0_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    return y0_arr

# ========================
# Config dataclass
# ========================
@dataclass
class BlockConfig:
    block_steps: int = 2048
    min_block_steps: int = 128
    max_block_steps: int = 8192
    grow_factor: float = 1.5
    shrink_factor: float = 0.5
    target_iter: int = 6
    max_iter: int = 80
    tol: float = 1e-10
    verbose: bool = False
    # --- new parameters for stability ---
    picard_omega: float = 0.7       # initial damping factor
    omega_backtrack: float = 0.5    # shrink omega if error not reduced

# ========================
# Main solver
# ========================
def solve_caputo_blocked(
    F: Callable[[Array, Array], Array],
    alpha: float,
    y0: Array | float,
    T: float,
    N: int = 20000,
    cfg: Optional[BlockConfig] = None,
) -> Tuple[Array, Array]:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if N < 2:
        raise ValueError("N must be >= 2")
    if cfg is None:
        cfg = BlockConfig()

    y0_vec = _ensure_2d_y0(y0)
    d = y0_vec.size

    t = np.linspace(0.0, T, N + 1)
    h = t[1] - t[0]
    factor = (h ** alpha) / math.gamma(alpha + 1.0)

    y = np.zeros((N + 1, d), dtype=float)
    y[0, :] = y0_vec

    f_hist = []  # list of (block_f), each (L_block, d)

    M = int(cfg.block_steps)
    n0 = 0

    while n0 < N:
        M_cur = min(M, N - n0)
        if cfg.verbose:
            print(f"\n[Block] start={n0}, steps={M_cur}")

        m = np.arange(n0 + M_cur, dtype=float)
        b = (m + 1.0) ** alpha - m ** alpha

        if n0 > 0:
            f_prev = np.vstack(f_hist)
        else:
            f_prev = np.zeros((0, d))

        if n0 > 0:
            conv_tail = _fft_convolve_columns(b, f_prev)
            idx_start = n0
            idx_end = n0 + M_cur
            H = conv_tail[idx_start:idx_end, :]
        else:
            H = np.zeros((M_cur, d))

        # Initial guess: predictor using constant y_start RHS
        y_block = np.zeros((M_cur + 1, d))
        y_block[0, :] = y[n0, :]
        t_block = t[n0 : n0 + M_cur + 1]

        y_const = np.tile(y_block[0, :], (M_cur + 1, 1))
        f_pred = F(y_const, t_block)
        if f_pred.ndim == 1:
            f_pred = f_pred[:, None]
        f_pred = f_pred[:M_cur, :]
        conv_pred = _fft_convolve_columns(
            (np.arange(M_cur, dtype=float) + 1.0) ** alpha - (np.arange(M_cur, dtype=float)) ** alpha,
            f_pred,
        )
        y_block[1:, :] = y0_vec + factor * (H + conv_pred[:M_cur, :])

        b_loc = (np.arange(M_cur, dtype=float) + 1.0) ** alpha - (np.arange(M_cur, dtype=float)) ** alpha

        converged = False
        omega = cfg.picard_omega
        prev_err = np.inf

        for it in range(cfg.max_iter):
            f_loc = F(y_block, t_block)
            if f_loc.ndim == 1:
                f_loc = f_loc[:, None]
            f_loc = f_loc[:M_cur, :]

            conv_loc = _fft_convolve_columns(b_loc, f_loc)
            y_fp = np.empty_like(y_block)
            y_fp[0, :] = y_block[0, :]
            y_fp[1:, :] = y0_vec + factor * (H + conv_loc[:M_cur, :])

            # Damped Picard update
            y_new = y_block + omega * (y_fp - y_block)

            err = np.max(np.abs(y_new - y_block))
            if cfg.verbose:
                print(f"  iter {it+1:02d}: sup|Δy|={err:.3e}, omega={omega:.3f}")

            if err >= prev_err * 0.999:  # not improving
                omega *= cfg.omega_backtrack
                if omega < 1e-3:
                    break
                continue

            y_block = y_new
            prev_err = err

            if err < cfg.tol:
                converged = True
                break

        # adapt block size
        if converged and (it + 1) <= max(1, cfg.target_iter // 2):
            M = min(int(M * cfg.grow_factor), cfg.max_block_steps)
        elif not converged or (it + 1) > cfg.target_iter * 2:
            M = max(int(M * cfg.shrink_factor), cfg.min_block_steps)

        if not converged:
            if M_cur > max(8, cfg.min_block_steps):
                if cfg.verbose:
                    print("  Not converged — shrinking and retrying block.")
                M = max(int(M_cur * cfg.shrink_factor), cfg.min_block_steps)
                continue
            else:
                # fallback: bisect block
                if M_cur > 8:
                    if cfg.verbose:
                        print("  Not converged — bisecting block.")
                    M = max(8, M_cur // 2)
                    continue
                else:
                    raise RuntimeError("Picard iteration failed even after bisecting.")

        # Accept block
        y[n0 + 1 : n0 + M_cur + 1, :] = y_block[1:, :]
        f_block = F(y_block, t_block)
        if f_block.ndim == 1:
            f_block = f_block[:, None]
        f_block = f_block[:M_cur, :]
        f_hist.append(f_block)

        n0 += M_cur

    return t, y

# ========================
# Demo
# ========================
def _demo_vector():
    d = 2
    alpha = 0.6
    y0 = np.array([1.0, 0.5])
    T = 20.0
    N = 20000
    A = np.array([[-0.8, 0.3], [-0.2, -1.1]])

    def F(y, t):
        return y[:-1] @ A.T + np.column_stack((np.sin(t[:-1]), np.cos(t[:-1])))

    cfg = BlockConfig(block_steps=1024, min_block_steps=64, max_block_steps=4096,
                      grow_factor=1.6, shrink_factor=0.5, target_iter=6,
                      max_iter=80, tol=1e-10, verbose=True)

    t, Y = solve_caputo_blocked(F, alpha, y0, T, N, cfg)

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, Y[:,0], label='y1')
        plt.plot(t, Y[:,1], label='y2')
        plt.xlabel('t'); plt.ylabel('y')
        plt.legend(); plt.title('Enhanced Vector-valued Caputo FODE (blocked)')
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    _demo_vector()
