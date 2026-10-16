"""Central configuration for the sycophancy spectral-analysis pipeline.

Every knob the pipeline uses lives here. Stages read the same Config object, so
changing a value here changes it everywhere. CLI flags in main.py override
these; a YAML file can override them too (--config my.yaml).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ---------------- model ----------------
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "bfloat16"                # bfloat16 | float16 | float32
    device: str = "cuda"
    hf_token_env: str = "HF_TOKEN"         # env var holding the HF access token
    hf_cache_dir: Optional[str] = None     # set to a PVC path on Nautilus

    # ---------------- dataset ----------------
    dataset_name: str = "truthful_qa"
    dataset_config: str = "generation"
    dataset_split: str = "validation"
    max_questions: Optional[int] = None    # None = all (817); set small for a dry run

    # ---------------- prompts ----------------
    seed: int = 42
    wrong_answer_index: int = 0            # 0 = first listed incorrect answer
                                           # (§3.1 "most common incorrect answer")

    # ---------------- generation ----------------
    max_new_tokens: int = 128
    gen_batch_size: int = 8                # set to 1 for bit-exact greedy decoding
    agreement_window_chars: int = 100      # §3.1.1

    # ---------------- hidden states ----------------
    layers: List[int] = field(default_factory=lambda: [0, 7, 14, 21, 27])
    include_embedding: bool = False        # add the true embedding output as layer -1
                                           # (see README "Deviations", note D2)
    extract_batch_size: int = 8

    # ---------------- spectral basis ----------------
    eig_backend: str = "scipy"             # scipy | torch  (torch = GPU eigh, ~20x faster)
    pca_center: bool = False               # False = project raw h (as written in §3.4)
    bases: List[str] = field(default_factory=lambda: ["identity", "pca", "laplacian"])
    conditions: List[str] = field(default_factory=lambda: ["pressured", "neutral", "shift"])
    save_projections: bool = True

    # ---------------- measures ----------------
    alpha_floor_rel: float = 1e-12         # clip |h_hat|^2 at max*this before log-log fit

    # ---------------- eigenmode analysis ----------------
    bonferroni_alpha: float = 0.05
    n_top_modes: int = 20                  # modes carried into classification/interpretation
    eigenmode_layer: int = 27              # layer used for the deep-dive mode analysis

    # ---------------- classification ----------------
    lr_C: float = 1.0
    n_folds: int = 5
    cv_seed: int = 0

    # ---------------- io ----------------
    out_dir: str = "artifacts"
    overwrite: bool = False                # False = skip stages whose outputs exist

    # ---------------- derived ----------------
    def path(self, *parts: str) -> str:
        p = os.path.join(self.out_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def dir(self, *parts: str) -> str:
        p = os.path.join(self.out_dir, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    @property
    def all_layers(self) -> List[int]:
        return ([-1] if self.include_embedding else []) + list(self.layers)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(yaml_path: Optional[str] = None, **overrides) -> Config:
    """Build a Config from defaults, an optional YAML file, and keyword overrides."""
    base = {}
    if yaml_path:
        import yaml  # only needed if a config file is actually used
        with open(yaml_path) as f:
            base = yaml.safe_load(f) or {}
    base.update({k: v for k, v in overrides.items() if v is not None})

    valid = {f.name for f in dataclasses.fields(Config)}
    unknown = set(base) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**base)
