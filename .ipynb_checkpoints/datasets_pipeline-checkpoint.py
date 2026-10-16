"""End-to-end dataset pipeline for the four FINAL sycophancy benchmarks:

    MMLU · MedMCQA · ARC-Challenge · TruthfulQA-MC1

Everything that used to live in notebook cells now lives here so the pipeline is
reproducible from one file: load -> normalize to a canonical schema -> (MMLU)
stratified subsample -> per-row option shuffle until answer position is balanced
-> save eda_out/*_final.parquet. Those parquet files feed prompt_builder.py.

Canonical row schema (identical across datasets, consumed downstream):
    idx           int    stable within-dataset id (survives shuffle & sampling)
    question      str
    choices       list[str]
    answer_idx    int    position of the correct option in `choices`
    answer_letter str    ALPHA[answer_idx]
    n_options     int
    q_len_words   int
    category      str    MMLU subject / MedMCQA subject / TruthfulQA category / "" for ARC

Verified HF paths & schemas (checked 2026-10-06):
  MMLU            cais/mmlu (config "all", split test)          question, choices[4], subject, answer(int 0-3)
  MedMCQA         openlifescienceai/medmcqa (split validation)  question, opa-opd, subject_name, cop(int 0-3)
  ARC-Challenge   allenai/ai2_arc (config ARC-Challenge, test)  question, choices{text,label}, answerKey
  TruthfulQA-MC1  truthfulqa/truthful_qa (multiple_choice +      question, mc1_targets{choices,labels};
                  generation for category, split validation)    category joined from generation config

Dataset-specific facts baked in (confirmed against live data):
  - MedMCQA: test labels withheld -> use validation; keep choice_type=="single" only.
  - ARC: answerKey mixes 'A'-'D' and '1'-'4' -> normalise via the row's own label list;
         ~0.6% of rows are not 4-option -> dropped for uniformity (four_only=True).
  - TruthfulQA-MC1: raw lists the correct option FIRST (labels=[1,0,...]) and option
         counts VARY per row -> shuffling is mandatory; dense_range handles variable k.

Run:  python datasets_pipeline.py            # writes eda_out/*_final.parquet
Import: from datasets_pipeline import build_all_frames
"""
from __future__ import annotations

import hashlib
import os
import string
import warnings
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ALPHA = string.ascii_uppercase
SEED = 42
MMLU_N = 5000          # stratified subsample target for MMLU (set None to keep all)

os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))


# ===========================================================================
# Canonical schema helper + per-dataset adapters
# ===========================================================================
def add_meta(r: dict, idx: int) -> dict:
    ai = r.get("answer_idx")
    r["idx"] = int(idx)
    r["answer_letter"] = ALPHA[ai] if (ai is not None and 0 <= ai < len(ALPHA)) else None
    r["n_options"] = len(r["choices"]) if r.get("choices") else None
    r["q_len_words"] = len(r["question"].split()) if r.get("question") else 0
    return r


def load_mmlu() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = [add_meta({"question": e["question"], "choices": list(e["choices"]),
                      "answer_idx": int(e["answer"]), "category": e["subject"]}, i)
            for i, e in enumerate(ds)]
    return pd.DataFrame(rows)


def load_medmcqa() -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    ds = ds.filter(lambda x: x.get("choice_type", "single") == "single")
    rows = [add_meta({"question": e["question"],
                      "choices": [e["opa"], e["opb"], e["opc"], e["opd"]],
                      "answer_idx": int(e["cop"]), "category": e.get("subject_name", "")}, i)
            for i, e in enumerate(ds)]
    return pd.DataFrame(rows)


def load_arc_challenge(split: str = "test", four_only: bool = True) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    rows = []
    for i, e in enumerate(ds):
        labels = list(e["choices"]["label"])
        texts = list(e["choices"]["text"])
        if e["answerKey"] not in labels:        # a couple of historically malformed rows
            continue
        ai = labels.index(e["answerKey"])        # robust to 'A'-'D' or '1'-'4'
        rows.append(add_meta({"question": e["question"], "choices": texts,
                              "answer_idx": ai, "category": ""}, i))
    df = pd.DataFrame(rows)
    if four_only:
        df = df[df["n_options"] == 4].reset_index(drop=True)
    return df


def _reduce_to_k(choices, ai, k_max, seed, idx):
    """Cap an item at k_max options: keep the correct one + (k_max-1) distractors sampled
    deterministically per item (fixed hash seed). Returns (choices, answer_idx) with the
    correct option placed first; the pipeline's per-row shuffle later randomizes position.
    Items already <= k_max are returned unchanged (2- and 3-option rows are kept)."""
    if k_max is None or len(choices) <= k_max:
        return choices, ai
    h = int(hashlib.md5(f"{seed}|tqa|{idx}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    wrong = [j for j in range(len(choices)) if j != ai]
    keep_wrong = list(rng.choice(wrong, size=k_max - 1, replace=False))
    kept = [choices[ai]] + [choices[j] for j in keep_wrong]
    return kept, 0


def load_truthfulqa_mc1(reduce_to: int = 4, seed: int = SEED) -> pd.DataFrame:
    """Options from the multiple_choice config; category joined from the generation
    config on the question string (817/817 match confirmed).

    reduce_to=4 caps items at 4 options (keep gold + 3 sampled distractors, per the
    attached ICLR paper's rule); items with <=4 options are kept as-is. reduce_to=None
    keeps the native variable option count (2-13). NOTE: correct option sits at index 0
    after loading -> the finalize step shuffles it out either way."""
    from datasets import load_dataset
    mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation").to_pandas()
    gen = load_dataset("truthfulqa/truthful_qa", "generation", split="validation").to_pandas()
    cat_by_q = dict(zip(gen["question"].str.strip(), gen["category"]))
    rows = []
    for i, r in mc.iterrows():
        choices = list(r["mc1_targets"]["choices"])
        labels = list(r["mc1_targets"]["labels"])
        ai = labels.index(1)
        choices, ai = _reduce_to_k(choices, ai, reduce_to, seed, i)
        rows.append(add_meta({"question": r["question"], "choices": choices,
                              "answer_idx": ai,
                              "category": cat_by_q.get(r["question"].strip(), "")}, i))
    df = pd.DataFrame(rows)
    missing = df["category"].eq("").sum()
    if missing:
        print(f"  !! TruthfulQA: {missing} rows failed category join (left blank)")
    if reduce_to:
        print(f"  TruthfulQA reduced to <= {reduce_to} options: "
              f"n_options now {sorted(df['n_options'].unique())}")
    return df


# ===========================================================================
# Stratified subsample (MMLU) — balanced across categories, reproducible
# ===========================================================================
def stratified_sample(df: pd.DataFrame, n_total: int, by: str = "category",
                      seed: int = SEED, scheme: str = "uniform") -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    cats = sorted(df[by].unique())
    if scheme == "proportional":
        quota = (df[by].value_counts(normalize=True) * n_total).round().astype(int)
    else:
        quota = pd.Series(n_total // len(cats), index=cats)
    parts = [df[df[by] == c].sample(n=min(int(quota[c]), (df[by] == c).sum()),
                                    random_state=rng) for c in cats]
    out = pd.concat(parts)
    if len(out) < n_total:
        rem = df.drop(out.index)
        if len(rem):
            out = pd.concat([out, rem.sample(n=min(n_total - len(out), len(rem)),
                                             random_state=rng)])
    return out.sample(frac=1, random_state=rng).reset_index(drop=True)


# ===========================================================================
# Option-order shuffling + answer-position balancing
# ===========================================================================
def _shuffle_group(g: pd.DataFrame, seed: int) -> pd.DataFrame:
    """One per-row option shuffle; correctness preserved, answer_idx/letter remapped."""
    rng = np.random.RandomState(seed)
    nc, ni = [], []
    for _, r in g.iterrows():
        perm = rng.permutation(len(r["choices"]))
        nc.append([r["choices"][i] for i in perm])
        ni.append(int(list(perm).index(r["answer_idx"])))
    out = g.copy()
    out["choices"] = nc
    out["answer_idx"] = ni
    out["answer_letter"] = [ALPHA[i] for i in ni]
    return out


def dense_range(g: pd.DataFrame, min_presence: float = 0.5) -> float:
    """Answer-position skew over positions present in >=min_presence of rows, each
    normalized by how many rows actually offer that position. Handles variable k
    (needed for TruthfulQA) by ignoring sparse high positions."""
    n = len(g)
    ks = g["choices"].apply(len)
    fr = []
    for p in range(ks.max()):
        rows_with_p = int((ks > p).sum())
        if rows_with_p / n >= min_presence:
            fr.append((g["answer_idx"] == p).sum() / rows_with_p)
    return (max(fr) - min(fr)) if fr else 0.0


def rebalance(df: pd.DataFrame, max_range: float = 0.06, max_attempts: int = 100,
              seed0: int = SEED, label: str = "") -> pd.DataFrame:
    """Re-randomize option order until dense_range <= max_range (or keep the best)."""
    best, best_r = None, np.inf
    for a in range(max_attempts):
        cand = _shuffle_group(df, seed0 + a)
        r = dense_range(cand)
        if r < best_r:
            best, best_r = cand, r
        if r <= max_range:
            print(f"  {label}: balanced in {a+1} attempt(s), dense_range={r:.3f}")
            return best
    print(f"  {label}: best dense_range={best_r:.3f} after {max_attempts} attempts (accepted)")
    return best


# ===========================================================================
# Orchestration
# ===========================================================================
def build_all_frames(mmlu_n: int = MMLU_N, seed: int = SEED,
                     save_dir: str = "eda_out") -> Dict[str, pd.DataFrame]:
    print("Loading datasets ...")
    frames = {
        "MMLU": load_mmlu(),
        "MedMCQA": load_medmcqa(),
        "ARC-Challenge": load_arc_challenge(),
        "TruthfulQA-MC1": load_truthfulqa_mc1(),
    }
    for n, d in frames.items():
        print(f"  {n:16s} {len(d):>5} rows, {d['category'].nunique()} categories, "
              f"n_options {sorted(d['n_options'].unique())}")

    if mmlu_n and len(frames["MMLU"]) > mmlu_n:
        before = len(frames["MMLU"])
        frames["MMLU"] = stratified_sample(frames["MMLU"], mmlu_n, seed=seed)
        print(f"\nMMLU subsampled {before} -> {len(frames['MMLU'])} "
              f"(~{len(frames['MMLU'])//frames['MMLU']['category'].nunique()} per subject)")

    print("\nShuffling option order & balancing answer position (seed=%d):" % seed)
    for n in list(frames):
        frames[n] = rebalance(frames[n], max_range=0.06, seed0=seed, label=n)

    print("\nFinal answer-position skew (dense_range):")
    for n, d in frames.items():
        print(f"  {n:16s} dense_range={dense_range(d):.3f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stems = {"MMLU": "mmlu", "MedMCQA": "medmcqa",
                 "ARC-Challenge": "arc", "TruthfulQA-MC1": "truthfulqa"}
        summary = []
        for n, d in frames.items():
            d.to_parquet(f"{save_dir}/{stems[n]}_final.parquet", index=False)
            summary.append({"dataset": n, "rows": len(d),
                            "n_categories": d["category"].nunique(),
                            "opts_min": int(d["n_options"].min()),
                            "opts_max": int(d["n_options"].max()),
                            "answer_pos_range_dense": round(dense_range(d), 3)})
        pd.DataFrame(summary).to_csv(f"{save_dir}/dataset_summary.csv", index=False)
        print(f"\nsaved final (shuffled, balanced) frames to {save_dir}/*_final.parquet")
        print(pd.DataFrame(summary).to_string(index=False))

    return frames


if __name__ == "__main__":
    build_all_frames()
