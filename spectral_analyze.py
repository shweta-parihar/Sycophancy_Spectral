#!/usr/bin/env python
"""Spectral sycophancy analysis (§3.2-§3.7) over ALL models x datasets.

Consumes the precomputed generations from run_models.py:
    <runs-dir>/<model_tag>/hidden_states.npz   keys L{layer}_{neutral,pressured} (n,d) float16
    <runs-dir>/<model_tag>/responses.parquet   one row per experiment (labels + metadata)
    <runs-dir>/<model_tag>/config.json         layers, n_hidden_layers, hidden_size

For each (model, dataset) cell it reruns, REUSING the validated math in src/:
    §3.2.3  neutral-vs-pressured difference sanity check (states already extracted by hooks)
    §3.3    identity / PCA / Laplacian bases  (+ a RANDOM-orthonormal control, report §6.1)
    §3.4    l1/l2, Gini, alpha in every basis x condition x layer
    §3.5    eigenmode energy redistribution + condition decomposition at the last layer
    §3.6    single-feature AUC sweep + cross-validated logistic regression
    §3.7    mode<->property correlations, quartiles, per-category separation

Sycophancy label per model (matches the results EDA): base -> log-prob (_lp),
instruct -> generation (_gen).

Datasets analyzed: each dataset present in the parquet, plus a pooled "ALL" cell per model.

Outputs: <out-dir>/<model>/<dataset>/{features.parquet, results/*.csv, *.json, summary.json}
         <out-dir>/index.json   (every cell's headline numbers, for aggregation/report)

Run (on a node with the states; torch backend uses the GPU for the eigendecompositions):
    python spectral_analyze.py --runs-dir model_runs --out-dir spectral_artifacts --eig-backend torch
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config                                  # noqa: E402
from src import bases as B, measures as MS, eigenmodes as EM, classify as CL, interpret as IN  # noqa: E402
from src.utils import log, save_json, set_seed, timed      # noqa: E402

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------
# model-type inference + label selection (same rule as the results EDA)
# ---------------------------------------------------------------------------
def infer(tag: str):
    t = tag.lower()
    fam = ("Qwen" if "qwen" in t else "Llama" if "llama" in t else
           "Mistral" if "mistral" in t else "?")
    typ = "instruct" if "instruct" in t else "base"
    return fam, typ


def softmax_p(lp_json, letter):
    try: d = json.loads(lp_json)
    except (TypeError, ValueError): return np.nan
    if letter not in d: return np.nan
    ks = list(d); v = np.array([d[k] for k in ks], float)
    p = np.exp(v - v.max()); p /= p.sum()
    return float(p[ks.index(letter)])


def load_model_tables(runs_dir: str, tag: str):
    """Returns (df, states, layers, last_layer). Aligns parquet rows to npz rows by position."""
    base = os.path.join(runs_dir, tag)
    df = pd.read_parquet(os.path.join(base, "responses.parquet")).reset_index(drop=True)
    with np.load(os.path.join(base, "hidden_states.npz")) as z:
        states = {k: z[k].astype(np.float32) for k in z.files}
    n = len(df)
    for k, arr in states.items():
        assert arr.shape[0] == n, f"{tag}: {k} has {arr.shape[0]} rows but parquet has {n}"
    layers = sorted({int(k.split("_")[0][1:]) for k in states})
    cfgp = os.path.join(base, "config.json")
    last = (json.load(open(cfgp)).get("layers", layers)[-1] if os.path.exists(cfgp) else layers[-1])

    fam, typ = infer(tag)
    # LOGITS for all models (base AND instruct): argmax over option-letter logits (always defined).
    df["primary_syc"]         = df["sycophantic_lp"]
    df["primary_correct_neu"] = df["correct_neutral_lp"]
    df["primary_flip_any"]    = df["flipped_any_lp"]
    if "target_letter" not in df.columns and "pushed_letter" in df.columns:
        df = df.rename(columns={"pushed_letter": "target_letter"})
    # continuous sycophancy strength: probability mass moved onto the asserted option
    df["target_mass_shift"] = (
        [softmax_p(j, l) for j, l in zip(df["option_logprobs_pressured"], df["target_letter"])]
        - np.array([softmax_p(j, l) for j, l in zip(df["option_logprobs_neutral"], df["target_letter"])]))
    return df, states, layers, int(last)


# ---------------------------------------------------------------------------
# bases (reuse src.bases; add a random-orthonormal control)
# ---------------------------------------------------------------------------
def build_random_basis(d: int, seed: int):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return {"eigenvalues": np.arange(d, dtype=np.float64), "basis": Q.astype(np.float32)}


def build_bases_for_layer(cfg, states, layer, seed):
    X = np.concatenate([states[f"L{layer}_neutral"], states[f"L{layer}_pressured"]], axis=0)
    out = {}
    for name in cfg.bases:
        if name == "identity":
            out[name] = B.build_identity_basis(X.shape[1])
        elif name == "pca":
            out[name] = B.build_pca_basis(X, cfg.eig_backend)
        elif name == "laplacian":
            out[name] = B.build_laplacian_basis(X, cfg.eig_backend)
        elif name == "random":
            out[name] = build_random_basis(X.shape[1], seed + layer)
        else:
            raise ValueError(f"unknown basis {name}")
    return out


# ---------------------------------------------------------------------------
# one (model, dataset) cell
# ---------------------------------------------------------------------------
def run_cell(cfg, tag, dataset, df_cell, states_cell, layers, last, out_dir):
    fam, typ = infer(tag)
    y = df_cell["primary_syc"].to_numpy().astype(bool)
    n, npos = len(y), int(y.sum())
    cell_dir = os.path.join(out_dir, tag, dataset.replace("/", "_"))
    res_dir = os.path.join(cell_dir, "results"); os.makedirs(res_dir, exist_ok=True)
    log.info(f"[{tag}/{dataset}] n={n} syc+={npos} ({npos/max(n,1):.3f})  layers={layers} last={last}")

    # §3.2.3 difference sanity check
    diff = {str(L): float(np.linalg.norm(states_cell[f"L{L}_pressured"]-states_cell[f"L{L}_neutral"], axis=1).mean())
            for L in layers}
    save_json({"n": n, "n_syc": npos, "layers": layers, "mean_shift_l2_by_layer": diff},
              os.path.join(cell_dir, "diff_stats.json"))

    if npos < 2 or (n - npos) < 2:
        save_json({"skipped": "degenerate labels", "n": n, "n_syc": npos},
                  os.path.join(cell_dir, "summary.json"))
        log.warning(f"[{tag}/{dataset}] too few per class -> skipped")
        return {"model": tag, "family": fam, "type": typ, "dataset": dataset,
                "n": n, "n_syc": npos, "skipped": True}

    # §3.3 + §3.4  bases, projections, measures (wide feature frame)
    feats = {}
    lap_proj = {}                               # last-layer laplacian projections, per condition
    for L in layers:
        built = build_bases_for_layer(cfg, states_cell, L, cfg.seed)
        for name, bs in built.items():
            for cond in cfg.conditions:
                H = B.condition_states(states_cell, L, cond)
                Hhat = B.project(H, bs["basis"])
                for m, vals in MS.compute_measures(Hhat, cfg).items():
                    feats[f"{m}__L{L}__{name}__{cond}"] = vals
                if name == "laplacian" and L == last:
                    lap_proj[cond] = Hhat
        g = MS.geometry_features(states_cell[f"L{L}_neutral"], states_cell[f"L{L}_pressured"])
        for k, v in g.items():
            feats[f"{k}__L{L}"] = v
        if L == last:
            lap_eig = built["laplacian"]["eigenvalues"]
    F = pd.DataFrame(feats)

    # §3.5 eigenmodes (last layer, laplacian)
    if cfg.normalize_energy:                    # report §6.2 fix: energy fractions
        norm_proj = {c: lap_proj[c] / (np.linalg.norm(lap_proj[c], axis=1, keepdims=True)+1e-30)
                     for c in lap_proj}
        er = EM.energy_redistribution(norm_proj["pressured"], y, lap_eig, cfg.bonferroni_alpha)
    else:
        er = EM.energy_redistribution(lap_proj["pressured"], y, lap_eig, cfg.bonferroni_alpha)
    er["table"].to_csv(os.path.join(res_dir, "eigenmode_energy_table.csv"), index=False)
    save_json(er["summary"], os.path.join(res_dir, "energy_summary.json"))
    cd = EM.condition_decomposition(lap_proj, y, lap_eig, cfg.bonferroni_alpha, use_energy=False)
    cd.to_csv(os.path.join(res_dir, "condition_decomposition.csv"), index=False)
    modes = EM.top_modes(cd, "pressured", cfg.n_top_modes)
    for cond in cfg.conditions:
        for m in modes:
            F[f"mode{m}__L{last}__laplacian__{cond}"] = lap_proj[cond][:, m]

    # §3.6.1 single-feature sweep (incl. the random-basis control)
    sf = CL.sweep_single_features(F, y)
    sf.to_csv(os.path.join(res_dir, "single_feature_auc.csv"), index=False)

    # §3.6.2 combined logistic regression (11-feature cross-basis keeps the 3 report bases)
    report_bases = [b for b in ("identity", "pca", "laplacian") if b in cfg.bases]
    cross = [f"l1_l2__L{last}__{b}__{c}" for b in report_bases for c in cfg.conditions]
    cross += [f"cos_neutral_pressured__L{last}", f"shift_l2__L{last}"]
    sets = {
        "cross_basis_l1l2": cross,
        "cross_basis + top4 modes": cross + [f"mode{m}__L{last}__laplacian__pressured" for m in modes[:4]],
        "cross_basis + top6 modes": cross + [f"mode{m}__L{last}__laplacian__pressured" for m in modes[:6]],
        "all_l1l2_all_layers": [c for c in F.columns if c.startswith("l1_l2__")],
    }
    try:
        lr = CL.evaluate_feature_sets(F, y, sets, cfg)
        lr.to_csv(os.path.join(res_dir, "logreg_feature_sets.csv"), index=False)
        best_comb = float(lr["auc_mean"].max()); best_set = lr.iloc[0]["feature_set"]
    except Exception as e:                       # too few per class for StratifiedKFold etc.
        log.warning(f"[{tag}/{dataset}] combined LR skipped: {e}")
        best_comb, best_set = float("nan"), None

    # §3.7 interpretation (reuse src.interpret with an MCQ props frame)
    props = pd.DataFrame({
        "sycophantic": y,
        "category": df_cell["category"].values if "category" in df_cell else "",
        "target_mass_shift": df_cell["target_mass_shift"].values,
        "primary_correct_neu": df_cell["primary_correct_neu"].astype(float).values,
        "cos_neutral_pressured": F[f"cos_neutral_pressured__L{last}"].values,
        "shift_l2": F[f"shift_l2__L{last}"].values,
    })
    mc = IN.correlate_modes(lap_proj, modes[:8], props)
    mc.to_csv(os.path.join(res_dir, "mode_correlations.csv"), index=False)
    pd.concat([IN.quartile_analysis(lap_proj["pressured"], m, props) for m in modes[:5]],
              ignore_index=True).to_csv(os.path.join(res_dir, "quartiles.csv"), index=False)
    ca = IN.category_analysis(lap_proj["pressured"], modes[0], props)
    if len(ca): ca.to_csv(os.path.join(res_dir, "categories.csv"), index=False)

    # headline numbers for the aggregator / report
    def auc_of(measure, basis, cond, layer):
        r = sf[(sf.measure==measure)&(sf.basis==basis)&(sf.condition==cond)&(sf.layer==layer)]
        return float(r["auc"].iloc[0]) if len(r) else float("nan")
    depth = {int(L): auc_of("l1_l2", "laplacian", "pressured", L) for L in layers}
    basis_cmp = {b: auc_of("l1_l2", b, "pressured", last)
                 for b in cfg.bases}                 # laplacian vs identity vs pca vs RANDOM
    best_sf = sf.iloc[0].to_dict() if len(sf) else {}
    base_acc = float(df_cell["primary_correct_neu"].mean())
    syc_rate = npos / max(n, 1)
    summary = {
        "model": tag, "family": fam, "type": typ, "dataset": dataset, "d": int(states_cell[f"L{last}_neutral"].shape[1]),
        "n": n, "n_syc": npos, "syc_rate": syc_rate,
        "baseline_acc": base_acc,
        "syc_conditional": (syc_rate / base_acc if base_acc > 0 else float("nan")),
        "label_source": "lp",
        "last_layer": last, "layers": layers,
        "best_single_feature": {k: best_sf.get(k) for k in ("measure","basis","condition","layer","auc")},
        "l1l2_laplacian_pressured_auc_by_layer": depth,
        "l1l2_pressured_auc_by_basis_lastlayer": basis_cmp,
        "combined_best_auc": best_comb, "combined_best_set": best_set,
        "n_significant_modes": er["summary"]["n_significant"],
        "top_modes": modes[:10],
        "normalize_energy": bool(cfg.normalize_energy),
    }
    save_json(summary, os.path.join(cell_dir, "summary.json"))
    F.to_parquet(os.path.join(cell_dir, "features.parquet"), index=False)
    log.info(f"[{tag}/{dataset}] best single AUC={best_sf.get('auc',float('nan')):.3f} "
             f"({best_sf.get('measure')}/{best_sf.get('basis')}/{best_sf.get('condition')}/L{best_sf.get('layer')})  "
             f"laplacian vs random (l1/l2,pressured,last): "
             f"{basis_cmp.get('laplacian',float('nan')):.3f} vs {basis_cmp.get('random',float('nan')):.3f}  "
             f"combined={best_comb:.3f}")
    return summary


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="model_runs")
    ap.add_argument("--out-dir", default="spectral_artifacts")
    ap.add_argument("--models", nargs="+", default=None, help="model tags (default: autodiscover)")
    ap.add_argument("--datasets", nargs="+", default=None, help="restrict to these dataset values")
    ap.add_argument("--no-all", action="store_true", help="skip the pooled ALL cell per model")
    ap.add_argument("--bases", nargs="+", default=["identity", "pca", "laplacian", "random"])
    ap.add_argument("--conditions", nargs="+", default=["pressured", "neutral", "shift"])
    ap.add_argument("--eig-backend", default="scipy", choices=["scipy", "torch"])
    ap.add_argument("--normalize-energy", action="store_true", help="report §6.2: per-mode tests on energy fractions")
    ap.add_argument("--n-top-modes", type=int, default=20)
    ap.add_argument("--lr-C", type=float, default=1.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)

    cfg = Config(bases=a.bases, conditions=a.conditions, eig_backend=a.eig_backend,
                 n_top_modes=a.n_top_modes, lr_C=a.lr_C, n_folds=a.n_folds, seed=a.seed,
                 out_dir=a.out_dir)
    cfg.normalize_energy = a.normalize_energy

    tags = a.models or sorted(os.path.basename(os.path.dirname(p))
                              for p in glob.glob(os.path.join(a.runs_dir, "*", "responses.parquet")))
    if not tags:
        raise SystemExit(f"no responses.parquet under {a.runs_dir}/*/")
    log.info(f"models: {tags}")

    index = []
    for tag in tags:
        with timed(f"load {tag}"):
            df, states, layers, last = load_model_tables(a.runs_dir, tag)
        ds_values = a.datasets or sorted(df["dataset"].unique())
        cells = [(ds, df["dataset"] == ds) for ds in ds_values]
        if not a.no_all:
            cells.append(("ALL", pd.Series(True, index=df.index)))
        for ds, mask in cells:
            idx = np.where(mask.to_numpy())[0]
            if len(idx) == 0:
                continue
            df_cell = df.iloc[idx].reset_index(drop=True)
            states_cell = {k: v[idx] for k, v in states.items()}
            with timed(f"{tag}/{ds}"):
                try:
                    index.append(run_cell(cfg, tag, ds, df_cell, states_cell, layers, last, a.out_dir))
                except Exception as e:
                    log.error(f"[{tag}/{ds}] FAILED: {e}")
                    index.append({"model": tag, "dataset": ds, "error": str(e)})
        del states
    save_json(index, os.path.join(a.out_dir, "index.json"))
    log.info(f"wrote {os.path.join(a.out_dir, 'index.json')} ({len(index)} cells)")


if __name__ == "__main__":
    main()
