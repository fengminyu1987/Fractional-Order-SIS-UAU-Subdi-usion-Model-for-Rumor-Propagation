# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_fractional_all.py
import gc
import json
import os
import time

import networkx as nx
import numpy as np

from fast_caputo_solver import solve_caputo_blocked, BlockConfig

# ---------------------------
# 参数扫描范围
# ---------------------------
alphas = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
beta_Us = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

# baseline 固定参数
N = 1000
mu = 1.0
delta = 0.5
lambd = 0.3
gamma = 1 / 3
m_consciousness = 4
m_risk = 2
T = 20.0
N_steps = 2048 * 2

# 意识层参数扫描 (实验3)，仅 baseline α=0.8, βU=0.6
consciousness_scan = {
    "m_consciousness": [2, 4, 8],
    "lambd": [0.2, 0.3, 0.5],
    "delta": [0.3, 0.5, 0.7],
}


# ---------------------------
# 初值生成
# ---------------------------
def generate_initial_conditions(N, dominant_state=0, dominance=0.8):
    y0 = np.zeros(3 * N)
    for i in range(N):
        p = np.zeros(3)
        p[dominant_state] = dominance
        remaining = 1.0 - dominance
        r1, r2 = np.random.rand(2)
        s = r1 + r2
        other = [j for j in range(3) if j != dominant_state]
        p[other] = remaining * np.array([r1, r2]) / s
        y0[3 * i:3 * i + 3] = p
    return y0


# ---------------------------
# 方程右端
# ---------------------------
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
            e = np.zeros(N);
            e[i] = 1
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


# ---------------------------
# 主运行
# ---------------------------
def run_all(output_dir=_os.path.join(_DATA, "results_fractional")):
    os.makedirs(output_dir, exist_ok=True)

    # α–βU 网格 (实验2)
    for alpha in alphas:
        for beta_U in beta_Us:
            case_dir = os.path.join(output_dir, f"alpha_{alpha}_betaU_{beta_U}")
            if os.path.exists(os.path.join(case_dir, "y.npy")):
                print(f"Skip α={alpha}, βU={beta_U}, already done.")
                continue

            params = dict(N=N, mu=mu, delta=delta, lambd=lambd,
                          beta_U=beta_U, gamma=gamma, alpha=alpha,
                          m_consciousness=m_consciousness,
                          m_risk=m_risk, T=T, N_steps=N_steps)

            print(f"\n=== Running α={alpha}, βU={beta_U} ===")
            np.random.seed(42)
            Gc = nx.barabasi_albert_graph(N, m_consciousness, seed=42)
            Gr = nx.barabasi_albert_graph(N, m_risk, seed=43)
            A, B = nx.to_numpy_array(Gc), nx.to_numpy_array(Gr)

            y0 = generate_initial_conditions(N)
            F = make_F(A, B, params)
            cfg = BlockConfig(16, 4, 2048, tol=1e-4, verbose=True,
                              picard_omega=0.7, omega_backtrack=0.8)

            t0 = time.time()
            t, y = solve_caputo_blocked(F, alpha, y0, T, N_steps, cfg)
            print(f"Time {time.time() - t0:.2f}s")

            case_dir = os.path.join(output_dir, f"alpha_{alpha}_betaU_{beta_U}")
            os.makedirs(case_dir, exist_ok=True)
            np.save(os.path.join(case_dir, "t.npy"), t)
            np.save(os.path.join(case_dir, "y.npy"), y)

            del y, t, F
            gc.collect()

            with open(os.path.join(case_dir, "params.json"), "w") as f:
                json.dump(params, f, indent=2)

    # 意识层实验 (实验3)
    baseline_alpha, baseline_betaU = 0.8, 0.6
    for mc in consciousness_scan["m_consciousness"]:
        for lam in consciousness_scan["lambd"]:
            for d in consciousness_scan["delta"]:
                case_dir = os.path.join(output_dir, f"conscious_m{mc}_lam{lam}_delta{d}")
                if os.path.exists(os.path.join(case_dir, "y.npy")):
                    print(f"Skip m={mc}, lambda={lam}, delta={d}, already done.")
                    continue
                params = dict(N=N, mu=mu, delta=d, lambd=lam,
                              beta_U=baseline_betaU, gamma=gamma,
                              alpha=baseline_alpha,
                              m_consciousness=mc, m_risk=m_risk,
                              T=T, N_steps=N_steps)

                print(f"\n=== Running consciousness m={mc}, λ={lam}, δ={d} ===")
                np.random.seed(42)
                Gc = nx.barabasi_albert_graph(N, mc, seed=42)
                Gr = nx.barabasi_albert_graph(N, m_risk, seed=43)
                A, B = nx.to_numpy_array(Gc), nx.to_numpy_array(Gr)

                y0 = generate_initial_conditions(N)
                F = make_F(A, B, params)
                cfg = BlockConfig(16, 2, 2048, tol=1e-4, verbose=True,
                                  picard_omega=0.7, omega_backtrack=0.8)

                t0 = time.time()
                t, y = solve_caputo_blocked(F, baseline_alpha, y0, T, N_steps, cfg)
                print(f"Time {time.time() - t0:.2f}s")

                case_dir = os.path.join(output_dir, f"conscious_m{mc}_lam{lam}_delta{d}")
                os.makedirs(case_dir, exist_ok=True)
                np.save(os.path.join(case_dir, "t.npy"), t)
                np.save(os.path.join(case_dir, "y.npy"), y)
                with open(os.path.join(case_dir, "params.json"), "w") as f:
                    json.dump(params, f, indent=2)


if __name__ == "__main__":
    run_all()
