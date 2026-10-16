#!/usr/bin/env python
"""Aggregate all spectral cells (<out-dir>/index.json + per-cell summary.json) into tidy
tables for the results report. Produces:

  spectral_summary.csv        one row per (model, dataset): best single-feature AUC,
                              laplacian-vs-identity-vs-random l1/l2 AUC (last layer,
                              pressured), combined-classifier AUC, syc rate, n, n_syc.
  spectral_depth_curve.csv    long form: model x dataset x layer -> l1/l2 Laplacian
                              pressured AUC (the "emerges with depth" curve).
  spectral_basis_control.csv  model x dataset -> l1/l2 pressured AUC in each basis at the
                              last layer, incl. the RANDOM control (report §6.1 check:
                              if random ~= laplacian, the graph construction isn't the
                              thing carrying the signal).

Run:  python spectral_aggregate.py --out-dir spectral_artifacts
"""
from __future__ import annotations
import argparse, glob, json, os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="spectral_artifacts")
    a = ap.parse_args()

    summ_paths = glob.glob(os.path.join(a.out_dir, "*", "*", "summary.json"))
    rows, depth_rows, basis_rows = [], [], []
    for p in sorted(summ_paths):
        s = json.load(open(p))
        if s.get("skipped"):
            continue
        bsf = s.get("best_single_feature", {})
        basis_cmp = s.get("l1l2_pressured_auc_by_basis_lastlayer", {})
        n_cav = s.get("n_caved"); n_res = s.get("n_resisted")
        # clean P-vs-R caving rate: caved / (caved + resisted), ignoring other-flips entirely.
        caved_rate = (round(n_cav / (n_cav + n_res), 4)
                      if (n_cav is not None and n_res is not None and (n_cav + n_res) > 0) else None)
        rows.append({
            "model": s["model"], "family": s.get("family"), "type": s.get("type"),
            "dataset": s["dataset"], "d": s.get("d"), "n": s["n"], "n_syc": s["n_syc"],
            "population": s.get("population"), "include_other": s.get("include_other"),
            "label": s.get("label_source"),
            "n_caved": n_cav, "n_resisted": n_res, "n_other": s.get("n_other"),
            "n_eligible_total": s.get("n_eligible_total"),
            "caved_rate": caved_rate,                      # <- use THIS as the caving rate in eligible mode
            "syc_rate": round(s["syc_rate"], 4),
            "baseline_acc": (round(s["baseline_acc"], 4) if s.get("baseline_acc") is not None else None),
            "syc_conditional": (round(s["syc_conditional"], 4) if s.get("syc_conditional") is not None else None),
            "best_sf_auc": bsf.get("auc"), "best_sf_measure": bsf.get("measure"),
            "best_sf_basis": bsf.get("basis"), "best_sf_cond": bsf.get("condition"),
            "best_sf_layer": bsf.get("layer"),
            "l1l2_laplacian": basis_cmp.get("laplacian"),
            "l1l2_identity": basis_cmp.get("identity"),
            "l1l2_pca": basis_cmp.get("pca"),
            "l1l2_random": basis_cmp.get("random"),
            "lap_minus_identity": (basis_cmp.get("laplacian") or 0) - (basis_cmp.get("identity") or 0)
                                  if basis_cmp.get("laplacian") is not None else None,
            "lap_minus_random": (basis_cmp.get("laplacian") or 0) - (basis_cmp.get("random") or 0)
                                if basis_cmp.get("laplacian") is not None else None,
            "combined_auc": s.get("combined_best_auc"), "combined_set": s.get("combined_best_set"),
            "n_sig_modes": s.get("n_significant_modes"),
        })
        for L, auc in (s.get("l1l2_laplacian_pressured_auc_by_layer") or {}).items():
            depth_rows.append({"model": s["model"], "dataset": s["dataset"],
                               "layer": int(L), "auc": auc})
        for b, auc in basis_cmp.items():
            basis_rows.append({"model": s["model"], "dataset": s["dataset"],
                               "basis": b, "l1l2_pressured_auc_lastlayer": auc})

    out = a.out_dir
    pd.DataFrame(rows).sort_values(["dataset", "family", "type"]).to_csv(
        os.path.join(out, "spectral_summary.csv"), index=False)
    pd.DataFrame(depth_rows).to_csv(os.path.join(out, "spectral_depth_curve.csv"), index=False)
    pd.DataFrame(basis_rows).to_csv(os.path.join(out, "spectral_basis_control.csv"), index=False)
    print(f"wrote spectral_summary.csv ({len(rows)} cells), "
          f"spectral_depth_curve.csv ({len(depth_rows)} rows), "
          f"spectral_basis_control.csv ({len(basis_rows)} rows) in {out}/")
    if rows:
        df = pd.DataFrame(rows)
        print("\n=== headline: Laplacian vs identity/pca/random (l1/l2, last layer, pressured) ===")
        cols = ["model","dataset","population","label","n_caved","n_resisted","caved_rate",
                "l1l2_laplacian","l1l2_identity","l1l2_pca","l1l2_random",
                "lap_minus_identity","lap_minus_random","combined_auc"]
        print(df[[c for c in cols if c in df.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
