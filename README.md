# Fractional-order SIS-UAU subdiffusion model — reproducibility package

Code and results supporting

> Q. Li, J. Tang, J. Kurths, M. Perc, and M. Feng, "Fractional-order SIS-UAU
> subdiffusion model for rumor propagation," *IEEE Transactions on Systems,
> Man, and Cybernetics: Systems*, 2026, DOI 10.1109/TSMC.2026.3741903.

All experiments are integrated Caputo fractional-order ODE models on duplex
networks, solved with a block-adaptive FFT solver (`src/fast_caputo_solver.py`)
and an integer-order reference solver / Monte Carlo simulator
(`src/integer_order_solver.py`, `src/monte_carlo_simulator.py`).

## Environment

Tested with Python 3.11.2 (Windows). Install the pinned dependencies:

```
pip install -r requirements.txt
```

Every experiment script can be executed from any working directory, e.g.

```
python experiments/fig4_fractional_dynamics/run_experiment1.py
```

The scripts add `src/` and their own folder to `sys.path` automatically, read
reference data from `data/...`, and write figures into `figures/` (or into
their experiment folder under `data/...`, exactly as in the original study).

## Contents

```
src/           engine: Caputo FFT solver (fractional), integer-order solver, Monte Carlo
experiments/   one folder per paper figure / experiment (see mapping below)
tests/         solver verification against the analytic Mittag-Leffler solution
figures/       final figure PDFs used in the paper and the Supplementary Material
figures/png/   matching PNG versions
data/          reference data: raw scans, cached results, summaries, npy caches
```

## Figure / result mapping

Main manuscript:

| Figure | File(s) in `figures/` | Producing script | Reference data |
|---|---|---|---|
| Fig. 3(a) | `comprehensive_comparison.pdf` | `experiments/fig3_integer_validation/run_comparison_experiment.py` | `data/comparison_results.pkl` |
| Fig. 3(b) | `parameter_analysis.pdf` | same script | same |
| Fig. 4(a),(b) | `experiment1_fractional_results.pdf`, `baseline_alpha_comparison.pdf` | `experiments/fig4_fractional_dynamics/run_experiment1.py` | `data/results_experiment1/` (npz caches; re-simulated if absent) |
| Fig. 5(a) | `heatmap_exp2_us_as_ai.pdf` | `experiments/fig5_alpha_beta_scan/plot_exp2_heatmap.py` | `data/results_fractional/` |
| Fig. 5(b)-(g) | `betaU_0.3 ... betaU_1.0_trajectories.pdf` | `experiments/fig5_alpha_beta_scan/analyze_experiment2.py` (α-legend in the β^U = 0.3 panel only, as in the manuscript) | same |
| Fig. 5 raw scan | — | `experiments/fig5_alpha_beta_scan/run_experiment2_scan.py` (produced `data/results_fractional/`, 36 cases) | — |
| Figs. 1–2 | schematic illustrations (not code-generated) | — | — |

Supplementary Material:

| Result | File(s) in `figures/` | Producing script | Reference data |
|---|---|---|---|
| Fig. S1 (sensitivity) | `sensitivity_tornado.pdf` | `experiments/sm_s1_sensitivity/sensitivity_analysis.py` | `data/results_fractional/` + `data/results_experiment3/` (no new simulations) |
| Fig. S2 (global stability) | `global_stability_free.pdf`, `global_stability_endemic.pdf` | `experiments/sm_s2_global_stability/regenerate_gs_figure.py` | `data/global_stability/gs_frac_*.npy` (caches) |
| Fig. S2 caches / integer reference | — | `run_gs_frac_endemic.py`, `run_global_stability_frac2.py`, `run_global_stability_int.py` (same folder) | `data/global_stability/` |
| Fig. S3 (convergence to α→1) | `convergence_alpha_to_1_with_error.pdf` | `experiments/sm_s3_convergence/run_experiment_extra_alpha_to_1.py` | `data/results_fractional/` |
| Fig. S4 (bifurcation) | `bifurcation_diagram_v3_new.pdf` | `experiments/sm_s4_bifurcation/plot_bifurcation.py` | `data/results_bifurcation/`, `data/results_bifurcation_frac/` |
| Fig. S4 raw data | — | `experiments/sm_s4_bifurcation/run_bifurcation.py` | — |
| Fig. S5(a) | `experiment3_3d_multilayer_tall.pdf` | `experiments/sm_s5_awareness/run_experiment3_enhanced.py` (+ `multi_layer_3d_plot.py`) | `data/results_experiment3/` |
| Fig. S5(b) | `consciousness_impact_summary.pdf` | `experiments/sm_s5_awareness/run_experiment3.py` | same (`run_experiment3.py --run` re-simulates the 300 cases) |
| Fig. S6 (network topology) | `experiment4_enhanced_comparison.pdf` | `experiments/sm_s6_topology/run_experiment4.py` | `data/results_experiment4/` (npz caches; re-simulated if absent) |
| Remark 4 (iii) thresholds | — | `experiments/paper_numbers/verify_thresholds.py` | BA duplex network built in the script |
| Remark 4 (iv) dissipativity ratio | — | `experiments/paper_numbers/verify_dissipativity.py` | same |
| §11.4 numbers (ρ^AI** = 0.2301, R0 = 2.07) | — | `experiments/sm_s2_global_stability/run_global_stability_int.py` | — |

## Data inventory

Included in this archive (large items are stored in the zip, not in the
browsable folder):

| Folder | Size | Content |
|---|---|---|
| `data/results_fractional/` | ≈ 5.8 GB | raw α×βU scan (36 cases; `t.npy`, `y.npy`, `params.json`) |
| `data/results_experiment1/` | ≈ 261 MB | Fig. 4 simulation caches (`results.npz`, summary, figures) |
| `data/results_experiment4/` | ≈ 424 MB | Fig. S6 simulation caches (9 topology combinations) |
| `data/results_experiment3/` | ≈ 10 MB | Fig. S5 data (per-case JSON, result matrices, figures) |
| `data/results_bifurcation/`, `data/results_bifurcation_frac/` | ≈ 35 MB | Fig. S4 data |
| `data/analysis_experiment2/` | < 1 MB | Fig. 5(b)-(g) outputs + summary |
| `data/global_stability/` | < 1 MB | Fig. S2 fractional-trajectory caches + verification outputs |
| `data/comparison_results.pkl`, `data/misc/` | ≈ 8 MB | Fig. 3 Monte Carlo/IBMFA cache, legacy caches |

## Verification performed when packaging (2026-10-14)

| Check | Result |
|---|---|
| `python -m py_compile` on all 25 scripts | pass |
| `tests/test_solver_vs_mittag_leffler.py` (solver vs analytic Mittag-Leffler) | max abs. error 1.1e-04 |
| `verify_thresholds.py` | β^U(R̄0=1) = 0.1259, β^U(R0=1) = 0.2894 (paper: 0.126 / 0.289) |
| `verify_dissipativity.py` | largest value found −0.1917 (paper: −0.19) |
| `regenerate_gs_figure.py` | regenerated Fig. S2; integer trajectories converge to ρ^AI = 0.2301 |
| `run_experiment3_enhanced.py` | regenerated Fig. S5(a) (identical layout to the manuscript) |
| Figure reconstruction check (19/19 figures, pixel level) | all identical, max MAE 0.0000 |

### Figure-by-figure reconstruction check

Every figure file shipped in this package (the regenerated script outputs under
`data/...` and the manuscript gallery copies under `figures/`) was rasterized
with Poppler at 100 dpi and compared, pixel by pixel, with the exact PDF
embedded in the submitted manuscript (mean absolute error on the 0–255 scale):

| Figure | Package file (script output) | MAE | Verdict |
|---|---|---|---|
| Fig. 3(a) | `figures/comprehensive_comparison.pdf` | 0.000 | identical |
| Fig. 3(b) | `figures/parameter_analysis.pdf` | 0.000 | identical |
| Fig. 4(a) | `data/results_experiment1/experiment1_fractional_results.pdf` | 0.000 | identical |
| Fig. 4(b) | `data/results_experiment1/baseline_alpha_comparison.pdf` | 0.000 | identical |
| Fig. 5(a) | `figures/heatmap_exp2_us_as_ai.pdf` | 0.000 | identical |
| Fig. 5(b) β^U=0.3 | `data/analysis_experiment2/betaU_0.3_trajectories.pdf` | 0.000 | identical |
| Fig. 5(c) β^U=0.4 | `data/analysis_experiment2/betaU_0.4_trajectories.pdf` | 0.000 | identical |
| Fig. 5(d) β^U=0.5 | `data/analysis_experiment2/betaU_0.5_trajectories.pdf` | 0.000 | identical |
| Fig. 5(e) β^U=0.6 | `data/analysis_experiment2/betaU_0.6_trajectories.pdf` | 0.000 | identical |
| Fig. 5(f) β^U=0.8 | `data/analysis_experiment2/betaU_0.8_trajectories.pdf` | 0.000 | identical |
| Fig. 5(g) β^U=1.0 | `data/analysis_experiment2/betaU_1.0_trajectories.pdf` | 0.000 | identical |
| Fig. S1 | `figures/sensitivity_tornado.pdf` | 0.000 | identical |
| Fig. S2 (disease-free) | `figures/global_stability_free.pdf` | 0.000 | identical |
| Fig. S2 (endemic) | `figures/global_stability_endemic.pdf` | 0.000 | identical |
| Fig. S3 | `figures/convergence_alpha_to_1_with_error.pdf` | 0.000 | identical |
| Fig. S4 | `figures/bifurcation_diagram_v3_new.pdf` | 0.000 | identical |
| Fig. S5(a) | `data/results_experiment3/experiment3_3d_multilayer_tall.pdf` | 0.000 | identical |
| Fig. S5(b) | `data/results_experiment3/consciousness_impact_summary.pdf` | 0.000 | identical |
| Fig. S6 | `data/results_experiment4/experiment4_enhanced_comparison.pdf` | 0.000 | identical |

All 19 comparisons are pixel-identical (maximum MAE over the whole set:
0.0000), i.e. re-running the packaged scripts on the packaged data reproduces
the published figures exactly. Small byte-level differences between two PDF
files carrying the same plot (when they exist) come only from PDF metadata
such as creation timestamps.

Note on Fig. 5(b): the manuscript prints the α-legend only in the first
trajectory panel (β^U = 0.3). `analyze_experiment2.py` accordingly calls
`ax.legend(loc='best', fontsize=24, framealpha=0.9)` for that panel only, so
the regenerated β^U = 0.3 panel carries the same legend as the published
figure; the remaining five panels are reproduced without a legend, exactly as
published.

The heavy scans were **not** re-executed during packaging; their full data are
included, so every figure can be regenerated without re-simulation, and the
scan scripts are provided for full regeneration:

| Script | Cost |
|---|---|
| `run_experiment2_scan.py` | 36 fractional solves, N=1000, T=20 (hours) |
| `run_bifurcation.py` | 40 integer + 16 fractional solves, N=50 (minutes–hours) |
| `run_experiment3.py --run` | 300 fractional solves, N=250 (hours) |
| `run_experiment4.py` | 9 fractional solves, N=500, T=20 (hours) |
| `run_comparison_experiment.py` | IBMFA + 15 Monte Carlo runs (minutes–hours) |

## Notes

* The two Remark 4 (iii) thresholds are evaluated on the BA duplex network
  realisation used for them in the study: N = 250, awareness layer m_A = 4
  (seed 42), risk layer m_B = 2 (seed 43), λ = 0.3, δ = 0.5, μ = 1, γ = 1/3.
* The dissipativity evaluation (≈ −0.19) follows the SM description: samples on
  the boundary face p^AI ≡ 0 plus local maximization, with d_i = e_i = 1, on a
  BA realisation (N = 250, m_A = 4, m_B = 2, both layers seed 1) at β^U = 0.6.
  Both scripts print the reproduced values and write JSON summaries to
  `data/paper_numbers/`.
* The archive is provided for review and reproducibility of the published
  results; please contact the corresponding authors before reuse.

## Removed legacy material (changelog)

Older code generations and one-off scripts were removed when packaging,
including: the previous `一阶连续导数模型` generation, earlier duplicates of
`run_experiment1.py` / `run_experiment3.py` / `run_experiment4.py`,
`run_experiment_1.py`, teaching/demo scripts (`example_usage.py`,
`exercise_complex_system.py`, `single_run.py`), legacy plotting scripts
(`visualization`, `plot_results.py`, `plot_global_stability.py`,
`inspect_legend.py`), old figure versions, `out_frac/`, `__pycache__` and
scratch caches (`4_22_1.pkl`, `simulation_results.pkl`). Renames applied here:
`main.py` → `run_experiment2_scan.py`, `test.py` → `plot_exp2_heatmap.py`,
`test_solver_ml.py` → `tests/test_solver_vs_mittag_leffler.py`.
