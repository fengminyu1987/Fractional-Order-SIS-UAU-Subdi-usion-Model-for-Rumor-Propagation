# --- reproducibility-package path bootstrap (added when packaging) ---
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PKG = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path[:0] = [_os.path.join(_PKG, "src"), _HERE]
_DATA = _os.path.join(_PKG, "data")
_FIGD = _os.path.join(_PKG, "figures")
# ---------------------------------------------------------------------

# run_gs_frac_endemic.py
import os, time
import numpy as np
import networkx as nx
from fast_caputo_solver import solve_caputo_blocked, BlockConfig

N = 250
m_consciousness, m_risk = 4, 2
SEED = 42
T = 20.0
LAM, DELTA, MU, GAMMA = 0.3, 0.5, 1.0, 1/3
ALPHA = 0.8

np.random.seed(SEED)
Gc = nx.barabasi_albert_graph(N, m_consciousness, seed=SEED)
Gr = nx.barabasi_albert_graph(N, m_risk, seed=SEED+1)
A = nx.to_numpy_array(Gc)
B = nx.to_numpy_array(Gr)

def make_ic(inf_frac, awr_frac, seed):
    rng = np.random.default_rng(seed)
    y0 = np.zeros(3*N)
    for i in range(N):
        p = np.array([1-inf_frac-awr_frac, awr_frac, inf_frac])
        p += rng.normal(0, 0.02, 3)
        p = np.clip(p, 0, 1)
        p = p / p.sum()
        y0[3*i:3*i+3] = p
    return y0

def make_F(A, B, params):
    N = params["N"]
    beta_U, beta_A = params["beta_U"], params["beta_U"]*params["gamma"]
    lambd, delta, mu = params["lambd"], params["delta"], params["mu"]
    def F(P, t):
        M = len(P)-1
        Pd = P[:-1,:].reshape(M, N, 3)
        BP, AP = np.zeros_like(Pd), np.zeros_like(Pd)
        for j in range(M):
            BP[j] = A @ Pd[j]; AP[j] = B @ Pd[j]
        M3 = np.array([[0,0,0],[delta,-delta,0],[0,mu,-mu]])
        Q = np.zeros((N, M, 3, 3))
        for i in range(N):
            e = np.zeros(N); e[i] = 1
            for j in range(M):
                s1 = e @ BP[j] @ np.array([[0],[1],[1]])
                s2 = e @ AP[j] @ np.array([[0],[0],[1]])
                M1 = s1 * np.array([[-lambd,lambd,0],[0,0,0],[0,0,0]])
                M2 = s2 * np.array([[-beta_U,0,beta_U],[0,-beta_A,beta_A],[0,0,0]])
                Q[i,j] = M3 + M1 + M2
        p = Pd.transpose(1,0,2)
        res = np.zeros((M, 3*N))
        for j in range(M):
            res[j] = np.concatenate([p[i,j] @ Q[i,j] for i in range(N)])
        return res
    return F

ics = [(0.01, 0.01), (0.05, 0.30), (0.15, 0.60), (0.30, 0.01)]
for idx, (i_, a_) in enumerate(ics):
    y0 = make_ic(i_, a_, 2000+idx)
    params = dict(N=N, mu=MU, delta=DELTA, lambd=LAM, beta_U=0.6, gamma=GAMMA,
                  alpha=ALPHA, m_consciousness=m_consciousness, m_risk=m_risk)
    F = make_F(A, B, params)
    cfg = BlockConfig(16, 4, 2048, tol=1e-4, picard_omega=0.7, omega_backtrack=0.8)
    t0 = time.time()
    tf, yf = solve_caputo_blocked(F, ALPHA, y0, T, 2048, cfg)
    rho = yf.reshape(-1, N, 3)[:, :, 2].mean(axis=1)
    np.save(_os.path.join(_DATA, 'global_stability', f'gs_frac_endemic_{idx}.npy'), np.vstack([tf, rho]))
    print(f"endemic IC({i_:.2f},{a_:.2f}) -> {rho[-1]:.6f} ({time.time()-t0:.0f}s)", flush=True)
print("Done", flush=True)
