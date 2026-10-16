# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------
"""Reproduce the dissipativity evaluation reported in the Supplementary Material
(Remark on the dissipativity condition (11.x), Section 11.2) and Remark 4 (iv)
of the main manuscript.

Procedure (as reported in the SM):
  * BA duplex network N=250, m_A=4, m_B=2 (both layers seed 1),
    lambda=0.3, delta=0.5, mu=1.0, gamma=1/3, beta^U=0.6;
  * ratio of the left-hand side of the dissipativity inequality (42) to
    sum_i (dx_i^2 + dy_i^2), with d_i = e_i = 1;
  * random search over the boundary face p^AI = 0 (5 x 10^4 samples) followed
    by local maximization from six starting points.

Expected largest value found: approximately -0.19 (manuscript: -0.19).
"""
import json

import networkx as nx
import numpy as np

N, MA, MB = 250, 4, 2
LAM, DELTA, MU, GAMMA = 0.3, 0.5, 1.0, 1.0 / 3.0
BETA_U = 0.6
BETA_A = GAMMA * BETA_U


def endemic_equilibrium(A, B):
    """Damped fixed-point iteration for the endemic equilibrium."""
    pI = np.full(N, 0.05)
    pU = 1 - pI
    for _ in range(100000):
        pA = 1 - pU
        MA_ = A @ pA
        MI = B @ pI
        pU_n = DELTA * (1 - pI) / (DELTA + LAM * MA_ + BETA_U * MI)
        pI_n = MI * (BETA_U * pU_n + BETA_A * pA) / (MU + BETA_A * MI)
        nU = 0.5 * pU_n + 0.5 * pU
        nI = 0.5 * pI_n + 0.5 * pI
        err = max(np.max(np.abs(nU - pU)), np.max(np.abs(nI - pI)))
        pU, pI = nU, nI
        if err < 1e-15:
            break
    return pU, 1 - pU - pI, pI


def main():
    rng = np.random.default_rng(4242)
    A = nx.to_numpy_array(nx.barabasi_albert_graph(N, MA, seed=1))
    B = nx.to_numpy_array(nx.barabasi_albert_graph(N, MB, seed=1))

    pUe, pASe, pAIe = endemic_equilibrium(A, B)
    pAe = pASe + pAIe
    print(f"equilibrium means: US={pUe.mean():.4f}  AS={pASe.mean():.4f}  AI={pAIe.mean():.4f}")

    def ratio_on_face(pAS):
        """Ratio on the boundary face p^AI = 0 (so M_I = 0 and G = 0)."""
        pU = 1 - pAS
        MA_ = A @ pAS
        H = LAM * MA_ * pU - DELTA * pAS
        dx = -pAIe
        dz = pAS - pAe
        den = np.sum(dx * dx) + np.sum(dz * dz)
        return (2 * np.sum(dz * H)) / den

    best = -9e9
    best_pAS = None
    for _ in range(50000):
        pAS = rng.random(N)
        r = ratio_on_face(pAS)
        if r > best:
            best = r
            best_pAS = pAS.copy()
    print(f"random search on face p^AI=0 (5e4 samples): best {best:.4f}")

    results = []
    for trial in range(6):
        pAS = best_pAS.copy() if trial == 0 else rng.random(N)
        cur = ratio_on_face(pAS)
        step = 0.03
        for _ in range(12000):
            d = rng.normal(size=N)
            d /= np.linalg.norm(d) + 1e-30
            t = np.clip(pAS + step * d, 0, 1)
            r = ratio_on_face(t)
            if r > cur:
                pAS, cur = t, r
                step *= 1.02
            else:
                step *= 0.999
        results.append(cur)
        print(f"  restart {trial}: {cur:.4f}")
    val = max(results)
    print(f"largest value found: {val:.4f}  (manuscript: -0.19)")

    out = _os.path.join(_DATA, "paper_numbers", "verify_dissipativity.json")
    _os.makedirs(_os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "network": {"N": N, "m_A": MA, "m_B": MB, "seed": 1},
            "params": {"lambda": LAM, "delta": DELTA, "mu": MU, "gamma": GAMMA, "beta_U": BETA_U},
            "procedure": "5e4 samples on face p^AI=0 + 6 local-maximization restarts (rng seed 4242)",
            "largest_value_found": val,
            "manuscript_value": -0.19,
        }, fh, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
