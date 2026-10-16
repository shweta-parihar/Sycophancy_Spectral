# Spectral analysis over all models × datasets (§3.2–§3.7)

Two new files run the spectral pipeline on the generations you already produced,
reusing your existing `src/` math unchanged.

## Where to put them
Drop both into your `sycophancy_spectral/` package root (next to `main.py` and `config.py`),
because they import `from config import Config` and `from src import ...`:

```
sycophancy_spectral/
  config.py  main.py  src/...
  spectral_analyze.py      <- new
  spectral_aggregate.py    <- new
```

## What they consume
For each model tag, from your run_models output:
```
model_runs/<tag>/hidden_states.npz   # L{layer}_{neutral,pressured}
model_runs/<tag>/responses.parquet   # labels + metadata (target_letter, option_logprobs_*)
model_runs/<tag>/config.json         # layers, n_hidden_layers, hidden_size
```
Model type is inferred from the tag (`instruct` in the name → instruct, else base), and the
sycophancy label follows your EDA rule: **base → `sycophantic_lp`, instruct → `sycophantic_gen`**.

## Run
```bash
# analysis (use torch backend on the A100 — much faster eigendecompositions)
python spectral_analyze.py --runs-dir model_runs --out-dir spectral_artifacts --eig-backend torch

# aggregate all cells into report tables
python spectral_aggregate.py --out-dir spectral_artifacts
```
Useful flags: `--models <tags>` (subset), `--datasets <names>` (subset), `--no-all`
(skip the pooled ALL cell), `--normalize-energy` (report §6.2 fix for the per-mode tests),
`--bases identity pca laplacian random` (drop `random` to reproduce the original 3-basis run).

## What a cell runs (per model × dataset, plus a pooled ALL cell per model)
- §3.2.3 neutral-vs-pressured shift sanity check → `diff_stats.json`
- §3.3 bases: identity, PCA, Laplacian, **random-orthonormal control**
- §3.4 ℓ1/ℓ2, Gini, α for every basis × condition × layer → `features.parquet`
- §3.5 eigenmode energy redistribution + condition decomposition (last layer, Laplacian)
       → `results/eigenmode_energy_table.csv`, `condition_decomposition.csv`, `energy_summary.json`
- §3.6.1 single-feature AUC sweep → `results/single_feature_auc.csv`
- §3.6.2 cross-validated logistic regression (11-feature cross-basis + mode sets)
       → `results/logreg_feature_sets.csv`
- §3.7 mode↔property correlations, quartiles, per-category separation
       → `results/{mode_correlations,quartiles,categories}.csv`
- `summary.json` — headline numbers for that cell

## Aggregated outputs (for the report)
- `spectral_summary.csv` — one row per cell: best single-feature AUC, ℓ1/ℓ2 AUC in
  Laplacian vs identity vs PCA vs **random** (last layer, pressured), combined-classifier AUC.
- `spectral_depth_curve.csv` — model × dataset × layer → Laplacian ℓ1/ℓ2 pressured AUC
  (the "emerges with depth" curve, now cross-model/dataset).
- `spectral_basis_control.csv` — the basis comparison incl. the random control.

## Two things to read in the aggregate, per your report's caveats
1. **Random-basis control (`lap_minus_random`).** If Laplacian ℓ1/ℓ2 AUC is not clearly
   above the random-orthonormal basis, "Laplacian beats coordinates" reduces to "any rotation
   beats coordinates" — this is the §6.1 check a reviewer will ask for.
2. **Laplacian spectral degeneracy still applies** (W=(C+1)/2 → near-complete graph). The
   ℓ1/ℓ2 and Gini headlines survive it (they summarize the whole vector), but per-mode claims
   are only trustworthy for the well-separated modes. `--normalize-energy` applies the §6.2 fix
   to the per-mode energy tests.
