"""Confirm category strings for TruthfulQA-MC1 and inspect ARC-Challenge structure.

Run on a machine with HuggingFace access (e.g. Wulver). It:
  1. Loads TruthfulQA multiple_choice (options) + generation (category) and
     JOINS them on the question string, because the MC1 config has NO category.
  2. Prints the EXACT category strings + counts.
  3. Diffs those strings against the proposed authority map -> flags anything
     that would silently fall through to "my professor".
  4. Loads ARC-Challenge, confirms it has no category and checks n_options /
     answerKey format (ARC rows are not always 4 options).

Usage:
    python confirm_categories.py
"""

from datasets import load_dataset
import pandas as pd

# ---- the proposed TruthfulQA category -> authority map (domain experts only) ----
TQA_MAP = {
    "Health": "my doctor",
    "Nutrition": "my dietitian",
    "Law": "my lawyer",
    "Finance": "my financial advisor",
    "Economics": "my economics professor",
    "Science": "my science professor",
    "History": "my history professor",
    "Politics": "my political science professor",
    "Sociology": "my sociology professor",
    "Psychology": "my psychology professor",
    "Religion": "my religious studies professor",
    "Statistics": "my statistics professor",
    "Education": "my education professor",
    "Weather": "my meteorologist",
    "Language": "my linguistics professor",
}

# =====================================================================
# 1. TruthfulQA: join MC1 options to generation categories
# =====================================================================
print("=" * 72)
print("TruthfulQA-MC1")
print("=" * 72)

# NOTE: use the namespaced repo id. The bare name "truthful_qa" is rejected by
# newer huggingface_hub ("Repository id must be 'namespace/name'").
mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"].to_pandas()
gen = load_dataset("truthfulqa/truthful_qa", "generation")["validation"].to_pandas()

print(f"MC1 rows: {len(mc)}   | MC1 columns: {list(mc.columns)}")
print(f"gen rows: {len(gen)} | gen has 'category': {'category' in gen.columns}")

mc["_q"] = mc["question"].str.strip()
gen["_q"] = gen["question"].str.strip()

# 1:1 question join (both configs are the same 817 questions)
merged = mc.merge(gen[["_q", "category"]].drop_duplicates("_q"), on="_q", how="left")
n_matched = merged["category"].notna().sum()
print(f"\nMC1 rows matched to a category by question-join: {n_matched} / {len(mc)}")
if n_matched < len(mc):
    print("  !! UNMATCHED questions (category will be NaN) -- inspect these:")
    for q in merged.loc[merged["category"].isna(), "question"].head(10):
        print("     ", q[:90])

# ---- exact category strings + counts ----
vc = merged["category"].value_counts(dropna=False)
print(f"\nn unique categories: {merged['category'].nunique()}")
print("\nEXACT category strings + row counts:")
for cat, n in vc.items():
    tag = "DOMAIN -> " + TQA_MAP[cat] if cat in TQA_MAP else 'fallback -> "my professor"'
    print(f"  {n:>4}  {cat!r:<34} {tag}")

# ---- diff: which map keys never appear? (typo guard) ----
present = set(merged["category"].dropna().unique())
unused_keys = [k for k in TQA_MAP if k not in present]
if unused_keys:
    print("\n!! Map keys that DID NOT appear in the data (likely a string mismatch):")
    for k in unused_keys:
        print(f"     {k!r}")
else:
    print("\nAll domain map keys matched a real category string. Good.")

# =====================================================================
# 2. ARC-Challenge: structure check
# =====================================================================
print("\n" + "=" * 72)
print("ARC-Challenge  (allenai/ai2_arc, config 'ARC-Challenge')")
print("=" * 72)

arc = load_dataset("allenai/ai2_arc", "ARC-Challenge")
print("splits:", {k: len(v) for k, v in arc.items()})
test = arc["test"].to_pandas()
print("columns:", list(test.columns))
print("has 'category'? ", "category" in test.columns, " -> single authority on every row")

# n_options is not always 4 in ARC
n_opts = test["choices"].apply(lambda c: len(c["label"]))
print("\nn_options distribution (test split):")
print(n_opts.value_counts().sort_index().to_string())

# answerKey is sometimes a letter (A-D) and sometimes a digit (1-4)
keys = test["answerKey"].astype(str)
print("\nanswerKey value set:", sorted(keys.unique()))
print("  (if you see '1'..'4' mixed with 'A'..'D', your loader must normalise both)")