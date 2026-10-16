#!/usr/bin/env python
"""Spectral sycophancy analysis (§3.2-§3.7) — SCENARIO-CONFIGURABLE driver.

Same math as spectral_analyze.py (reuses config.py + src/), but adds switches so you can
run different label / population scenarios and compare, WITHOUT re-running the models.
Reads the precomputed:
    <runs-dir>/<model_tag>/hidden_states.npz   keys L{layer}_{neutral,pressured} (n,d) float16
    <runs-dir>/<model_tag>/responses.parquet   one row per experiment (labels + metadata)
    <runs-dir>/<model_tag>/config.json         layers, hidden_size

Scenario knobs (all leave the graph as the existing (C+1)/2 Laplacian):
    --types        which model types to include     (default: instruct only)
    --label        gen | lp  answer used for correct/caved  (default: gen = generation)
    --population    all | eligible                   (default: eligible)
        eligible = rows parseable on BOTH prompts AND correct on neutral (the opportunity set),
                   built from the RESTRICTED rows (bases + measures + AUC all on this subset).
        positive = caved (moved to the pushed answer);  negative = resisted (stayed correct).
        --include-other  also puts "flipped-to-a-different-wrong" (O) rows in the negative class.
    --no-all       skip the pooled ALL cell per model

Example (instruct-only, generation label, eligible P-vs-R, GPU eigendecomp):
    python spectral_analyze_scenarios.py --runs-dir model_runs --out-dir spectral_eligible \
           --eig-backend torch
"""
from __future__ import annotations

import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:                                    # tqdm missing -> no-op bar
    def tqdm(x=None, **k):
        return x if x is not None else _Null()
    class _Null:
        def update(self, *a, **k): pass
        def set_postfix_str(self, *a, **k): pass
        def close(self): pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config                                  # noqa: E402
from src import bases as B, measures as MS, eigenmodes as EM, classify as CL, interpret as IN  # noqa: E402
from src.utils import log, save_json, set_seed, timed      # noqa: E402
from bases_signed import build_signed_laplacian_basis       # noqa: E402  (signed Laplacian)

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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


def load_model_tables(runs_dir: str, tag: str, label: str):
    """Returns (df, states, layers, last). Sets primary_* by `label` and adds
    population masks (_eligible/_caved/_resisted/_other) used for the 'eligible' scenario."""
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

    s = label                                            # "gen" or "lp"
    df["primary_syc"]         = df[f"sycophantic_{s}"].astype(bool)
    df["primary_correct_neu"] = df[f"correct_neutral_{s}"].astype(bool)
    df["primary_flip_any"]    = df[f"flipped_any_{s}"].astype(bool)
    if "target_letter" not in df.columns and "pushed_letter" in df.columns:
        df = df.rename(columns={"pushed_letter": "target_letter"})
    df["target_mass_shift"] = (
        [softmax_p(j, l) for j, l in zip(df["option_logprobs_pressured"], df["target_letter"])]
        - np.array([softmax_p(j, l) for j, l in zip(df["option_logprobs_neutral"], df["target_letter"])]))

    # ---- population masks ----
    # eligible = parseable on both prompts AND correct on neutral (opportunity set).
    # For label=lp parse flags don't apply (argmax always defined) -> only require correct-neutral.
    if s == "gen":
        par_both = df["neutral_parseable"].astype(bool) & df["pressured_parseable"].astype(bool)
    else:
        par_both = pd.Series(True, index=df.index)
    df["_eligible"] = par_both & df["primary_correct_neu"]
    # answer under pressure (parsed letter for gen, argmax for lp). Partition eligible by it so
    # caved / resisted / other are disjoint by construction (target_letter != answer_letter).
    pressed = df["pressured_parsed"] if s == "gen" else df["pressured_argmax"]
    df["_caved"]    = df["_eligible"] & (pressed == df["target_letter"])        # -> pushed answer
    df["_resisted"] = df["_eligible"] & (pressed == df["answer_letter"])        # stayed correct
    df["_other"]    = df["_eligible"] & ~df["_caved"] & ~df["_resisted"]        # flipped elsewhere
    return df, states, layers, int(last)


def build_random_basis(d: int, seed: int):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return {"eigenvalues": np.arange(d, dtype=np.float64), "basis": Q.astype(np.float32)}


def build_bases_for_layer(cfg, states, layer, seed):
    X = np.concatenate([states[f"L{layer}_neutral"], states[f"L{layer}_pressured"]], axis=0)
    out = {}
    for name in cfg.bases:
        if name == "identity":   out[name] = B.build_identity_basis(X.shape[1])
        elif name == "pca":      out[name] = B.build_pca_basis(X, cfg.eig_backend)
        elif name == "laplacian":out[name] = B.build_laplacian_basis(X, cfg.eig_backend)
        elif name == "signed":   out[name] = build_signed_laplacian_basis(X, cfg.eig_backend, getattr(cfg, "knn", None))
        elif name == "random":   out[name] = build_random_basis(X.shape[1], seed + layer)
        else: raise ValueError(f"unknown basis {name}")
    return out


def run_cell(cfg, tag, dataset, df_cell, states_cell, layers, last, out_dir, elig_counts):
    fam, typ = infer(tag)
    ycol = "_caved" if cfg.population == "eligible" else "primary_syc"
    y = df_cell[ycol].to_numpy().astype(bool)
    n, npos = len(y), int(y.sum())
    cell_dir = os.path.join(out_dir, tag, dataset.replace("/", "_"))
    res_dir = os.path.join(cell_dir, "results"); os.makedirs(res_dir, exist_ok=True)
    n_cav, n_res, n_oth = elig_counts["caved"], elig_counts["resisted"], elig_counts["other"]
    log.info(f"[{tag}/{dataset}] pop={cfg.population} n={n} pos={npos} | eligible: "
             f"caved={n_cav} resisted={n_res} other={n_oth}  last={last}")

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

    feats = {}; lap_proj = {}
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
            last_built = built                      # keep to read other bases' eigenvalues
    F = pd.DataFrame(feats)

    # spectrum-degeneracy diagnostic: fraction of eigenvalues within 1e-3 of lambda=1
    # (the old (C+1)/2 Laplacian had ~98% here -> near-complete graph -> basis ~= identity)
    def degen(eig, ref=1.0, tol=1e-3):
        e = np.asarray(eig, float)
        return {"frac_near_1": float(np.mean(np.abs(e - ref) < tol)), "eig_std": float(e.std())}
    eig_degen = {b: degen(last_built[b]["eigenvalues"])
                 for b in ("laplacian", "signed") if b in last_built}

    if cfg.normalize_energy:
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

    sf = CL.sweep_single_features(F, y)
    sf.to_csv(os.path.join(res_dir, "single_feature_auc.csv"), index=False)

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
    except Exception as e:
        log.warning(f"[{tag}/{dataset}] combined LR skipped: {e}")
        best_comb, best_set = float("nan"), None

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

    def auc_of(measure, basis, cond, layer):
        r = sf[(sf.measure==measure)&(sf.basis==basis)&(sf.condition==cond)&(sf.layer==layer)]
        return float(r["auc"].iloc[0]) if len(r) else float("nan")
    depth = {int(L): auc_of("l1_l2", "laplacian", "pressured", L) for L in layers}
    basis_cmp = {b: auc_of("l1_l2", b, "pressured", last) for b in cfg.bases}
    best_sf = sf.iloc[0].to_dict() if len(sf) else {}
    base_acc = float(df_cell["primary_correct_neu"].mean())     # ~1.0 in eligible mode
    syc_rate = npos / max(n, 1)                                 # caved / (P+R[+O]) in eligible mode
    summary = {
        "model": tag, "family": fam, "type": typ, "dataset": dataset,
        "d": int(states_cell[f"L{last}_neutral"].shape[1]),
        "population": cfg.population, "include_other": bool(cfg.include_other),
        "label_source": cfg.label,
        "n": n, "n_syc": npos, "n_eligible_total": n_cav + n_res + n_oth,
        "n_caved": n_cav, "n_resisted": n_res, "n_other": n_oth,
        "syc_rate": syc_rate, "baseline_acc": base_acc,
        "syc_conditional": (syc_rate / base_acc if base_acc > 0 else float("nan")),
        "last_layer": last, "layers": layers,
        "best_single_feature": {k: best_sf.get(k) for k in ("measure","basis","condition","layer","auc")},
        "l1l2_laplacian_pressured_auc_by_layer": depth,
        "l1l2_pressured_auc_by_basis_lastlayer": basis_cmp,
        "combined_best_auc": best_comb, "combined_best_set": best_set,
        "n_significant_modes": er["summary"]["n_significant"],
        "top_modes": modes[:10], "normalize_energy": bool(cfg.normalize_energy),
        "knn": getattr(cfg, "knn", None), "eig_degeneracy": eig_degen,
    }
    save_json(summary, os.path.join(cell_dir, "summary.json"))
    F.to_parquet(os.path.join(cell_dir, "features.parquet"), index=False)
    sg = eig_degen.get("signed", {}); lp = eig_degen.get("laplacian", {})
    log.info(f"[{tag}/{dataset}] l1/l2  signed={basis_cmp.get('signed',float('nan')):.3f} "
             f"lap={basis_cmp.get('laplacian',float('nan')):.3f} "
             f"pca={basis_cmp.get('pca',float('nan')):.3f} "
             f"id={basis_cmp.get('identity',float('nan')):.3f} "
             f"rnd={basis_cmp.get('random',float('nan')):.3f}  combined={best_comb:.3f} | "
             f"degeneracy frac_near_1: lap={lp.get('frac_near_1',float('nan')):.3f} "
             f"signed={sg.get('frac_near_1',float('nan')):.3f}")
    return summary


def pop_mask(df, population, include_other):
    if population == "all":
        return pd.Series(True, index=df.index)
    m = df["_caved"] | df["_resisted"]
    if include_other:
        m = m | df["_other"]
    return m


def dataset_values(runs_dir, tag):
    p = os.path.join(runs_dir, tag, "responses.parquet")
    return sorted(pd.read_parquet(p, columns=["dataset"])["dataset"].unique())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="model_runs")
    ap.add_argument("--out-dir", default="spectral_artifacts_signed")
    ap.add_argument("--models", nargs="+", default=None, help="model tags (default: autodiscover)")
    ap.add_argument("--types", nargs="+", default=["instruct"], help="model types to include (default: instruct)")
    ap.add_argument("--datasets", nargs="+", default=None, help="restrict to these dataset values")
    ap.add_argument("--label", default="gen", choices=["gen", "lp"], help="answer label for correct/caved")
    ap.add_argument("--population", default="eligible", choices=["all", "eligible"])
    ap.add_argument("--include-other", action="store_true", help="put other-flip (O) rows in the negative class")
    ap.add_argument("--no-all", action="store_true", help="skip the pooled ALL cell per model")
    ap.add_argument("--bases", nargs="+", default=["identity", "pca", "laplacian", "signed", "random"])
    ap.add_argument("--knn", type=int, default=None, help="k-NN-sparsify the signed graph on |C| (default: dense)")
    ap.add_argument("--conditions", nargs="+", default=["pressured", "neutral", "shift"])
    ap.add_argument("--eig-backend", default="scipy", choices=["scipy", "torch"])
    ap.add_argument("--normalize-energy", action="store_true")
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
    cfg.population = a.population; cfg.include_other = a.include_other; cfg.label = a.label
    cfg.knn = a.knn

    tags = a.models or sorted(os.path.basename(os.path.dirname(p))
                              for p in glob.glob(os.path.join(a.runs_dir, "*", "responses.parquet")))
    tags = [t for t in tags if infer(t)[1] in a.types]
    if not tags:
        raise SystemExit(f"no matching runs under {a.runs_dir}/*/ for types={a.types}")
    log.info(f"models: {tags}  | label={a.label} population={a.population} include_other={a.include_other}")

    # ---- progress bar total (count cells from the light parquet read) ----
    total = 0
    ds_by_tag = {}
    for t in tags:
        dv = a.datasets or dataset_values(a.runs_dir, t)
        ds_by_tag[t] = dv
        total += len(dv) + (0 if a.no_all else 1)
    bar = tqdm(total=total, desc="cells")

    index = []
    for tag in tags:
        with timed(f"load {tag}"):
            df, states, layers, last = load_model_tables(a.runs_dir, tag, a.label)
        pm = pop_mask(df, a.population, a.include_other)
        cells = [(ds, (df["dataset"] == ds) & pm) for ds in ds_by_tag[tag]]
        if not a.no_all:
            cells.append(("ALL", pm.copy()))
        for ds, mask in cells:
            bar.set_postfix_str(f"{tag}/{ds}")
            ds_mask = pd.Series(True, index=df.index) if ds == "ALL" else (df["dataset"] == ds)
            elig_counts = {"caved":    int((ds_mask & df["_caved"]).sum()),
                           "resisted": int((ds_mask & df["_resisted"]).sum()),
                           "other":    int((ds_mask & df["_other"]).sum())}
            idx = np.where(mask.to_numpy())[0]
            if len(idx) == 0:
                log.warning(f"[{tag}/{ds}] 0 rows after population filter -> skipped")
                bar.update(1); continue
            df_cell = df.iloc[idx].reset_index(drop=True)
            states_cell = {k: v[idx] for k, v in states.items()}
            with timed(f"{tag}/{ds}"):
                try:
                    index.append(run_cell(cfg, tag, ds, df_cell, states_cell, layers, last, a.out_dir, elig_counts))
                except Exception as e:
                    log.error(f"[{tag}/{ds}] FAILED: {e}")
                    index.append({"model": tag, "dataset": ds, "error": str(e)})
            bar.update(1)
        del states
    bar.close()
    save_json(index, os.path.join(a.out_dir, "index.json"))
    log.info(f"wrote {os.path.join(a.out_dir, 'index.json')} ({len(index)} cells)")


if __name__ == "__main__":
    main()
