# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------
"""Reproduce the two thresholds quoted in Remark 4 (iii) of the manuscript.

BA duplex network of the theory/global-stability experiments:
    N = 250, awareness layer m_A = 4 (seed 42), risk layer m_B = 2 (seed 43)
    lambda = 0.3, delta = 0.5, mu = 1.0, gamma = 1/3

  (1) Rbar_0 = beta^U * rho(B) / mu = 1          -> beta^U ~= 0.126
  (2) R_0    = rho(diag(p^US* beta^U + p^AS* beta^A) B) / mu = 1
                                                  -> beta^U ~= 0.289
where p^US* = 1 - p^AS* and p^AS* is the awareness distribution at the
rumor-free equilibrium.
"""
import json

import networkx as nx
import numpy as np

N, MA, MB = 250, 4, 2
LAM, DELTA, MU, GAMMA = 0.3, 0.5, 1.0, 1.0 / 3.0


def main():
    A = nx.to_numpy_array(nx.barabasi_albert_graph(N, MA, seed=42))
    B = nx.to_numpy_array(nx.barabasi_albert_graph(N, MB, seed=43))

    rho_B = float(np.max(np.abs(np.linalg.eigvals(B))))
    beta_rbar = MU / rho_B

    M = np.full(N, 0.1)
    for _ in range(100000):
        p_aware = LAM * M / (DELTA + LAM * M)
        M_new = A @ p_aware
        if np.max(np.abs(M_new - M)) < 1e-15:
            M = M_new
            break
        M = M_new
    p_AS = LAM * M / (DELTA + LAM * M)
    p_US = DELTA / (DELTA + LAM * M)

    D = np.diag(p_US + GAMMA * p_AS)
    rho_DB = float(np.max(np.abs(np.linalg.eigvals(D @ B))))
    beta_r0 = MU / rho_DB

    print(f"Network: BA duplex, N={N}, m_A={MA} (seed 42), m_B={MB} (seed 43)")
    print(f"rho(B)  = {rho_B:.4f}  ->  beta^U at Rbar_0 = 1 : {beta_rbar:.4f}  (manuscript: 0.126)")
    print(f"rho(D B) = {rho_DB:.4f} ->  beta^U at R_0 = 1    : {beta_r0:.4f}  (manuscript: 0.289)")
    ok1 = abs(beta_rbar - 0.126) <= 0.002
    ok2 = abs(beta_r0 - 0.289) <= 0.002
    print("PASS" if (ok1 and ok2) else "CHECK values")

    out = _os.path.join(_DATA, "paper_numbers", "verify_thresholds.json")
    _os.makedirs(_os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "network": {"N": N, "m_A": MA, "seed_A": 42, "m_B": MB, "seed_B": 43},
            "params": {"lambda": LAM, "delta": DELTA, "mu": MU, "gamma": GAMMA},
            "rho_B": rho_B,
            "beta_Rbar0_equals_1": beta_rbar,
            "rho_DB": rho_DB,
            "beta_R0_equals_1": beta_r0,
            "manuscript_values": {"beta_Rbar0_equals_1": 0.126, "beta_R0_equals_1": 0.289},
        }, fh, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
