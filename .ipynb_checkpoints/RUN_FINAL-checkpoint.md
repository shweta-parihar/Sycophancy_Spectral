# Final pipeline — what to run on Wulver

Four files go into your package dir (next to `config.py`, `src/`, `bases_signed.py`,
`bases_extra.py`, `measures_extra.py`, `make_dataset_table.py`):

- `probes_full.py`       (NEW name; replaces old probes.py — now per-layer curves + CV single-feature + L1 sparse)
- `localize_signed.py`   (NEW — the four "where does sycophancy live" analyses, signed basis)
- `spectral_analyze_full.py`   (REPLACES — wires in per-layer probes, CV depth curves, localization, fixed combined set)
- `spectral_aggregate_full.py` (REPLACES — emits the new depth + localization tables)

Everything else you already have is unchanged.

## 1. Dataset table (fast, if you haven't already)
    python make_dataset_table.py --runs-dir model_runs

## 2. Main analysis (instruct-only, generation label, eligible caved-vs-resisted)
    python spectral_analyze_full.py \
        --runs-dir model_runs \
        --out-dir  spectral_artifacts_full \
        --eig-backend torch          # drop this if no GPU; scipy is the default

This is ONE consolidated run. It produces, per (model, dataset) cell:
  - all concentration + frequency measures, all 7 bases (incl. signed)
  - per-layer probe curves (diff-means + logreg, every layer, all 3 conditions)
  - CV single-feature depth curves for l1/l2 and spectral-spread on the signed basis
  - the fixed signed conc+freq combined set (headline combined classifier)
  - localization on the signed basis (mode-frequency, band-energy, probe-direction, sparse probe)

Defaults you can leave alone: `--label gen --population eligible --n-bands 5 --n-folds 5`.
Add `--knn 10` only if you want the k-NN-sparsified signed graph (dense is the default).

## 3. Aggregate to tidy CSVs
    python spectral_aggregate_full.py --out-dir spectral_artifacts_full

New tables it writes (on top of the old spectral_summary / depth_curve / basis_control):
  - `probe_depth_curve.csv`          model×dataset×layer×condition -> diff-means & logreg AUC
  - `single_feature_depth_curve.csv` model×dataset×layer -> l1/l2 & spec-spread CV AUC (signed)
  - `loc_mode_frequency.csv`         per signed eigenmode: eigenvalue, Cohen's d, CV AUC
  - `loc_band_energy.csv`            frequency band: caved vs resisted energy + Cohen's d
  - `loc_probe_direction.csv`        probe direction energy binned over graph frequency
  - `loc_sparse_probe.csv`           L1 strength -> (n_nonzero dims, AUC) + full-h ceiling

## 4. Zip and send back
    zip -r final_results.zip spectral_artifacts_full dataset_info.csv
Send me `final_results.zip` and I'll build the report.

Runtime is the same order as your last full run — the per-layer probes and localization add a
few seconds per cell, nothing structural.
