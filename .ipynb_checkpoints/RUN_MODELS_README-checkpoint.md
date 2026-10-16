# Running the 6 models on the Jupyter GPU terminal (A100 80GB)

`run_models.py` runs ONE model over all prompts (both conditions) and saves, per model:
- `hidden_states.npz`  — last-token states at blocks {0,7,14,21,27}, keys `L{layer}_{neutral,pressured}`, float16
- `responses.parquet`  — one row per experiment: generated text, parsed answer, option log-probs, both label types
- `config.json`        — run summary (sycophancy rates, unparseable rate)

This is the format the spectral pipeline consumes (same as the preliminary work).

---

## 0. HuggingFace token + cache

```bash
export HF_TOKEN=hf_xxxxxxxx                 # your token; accept Llama's license on HF first
export HF_HOME=$HOME/work/hf_cache          # cache weights on your PVC, not ephemeral disk
```

## 1. Batch size for your GPU

You have an **A100 80GB**, so:
- `--batch-size 32`  → safe default (recommended)
- `--batch-size 64`  → fine on 80GB, faster; drop to 32 if you ever see OOM
Check live memory any time from a second terminal with `nvidia-smi`.

## 2. tmux — keep the run alive if your browser disconnects

Long runs should live inside tmux so a dropped browser/Jupyter tab doesn't kill them.

```bash
tmux new -s syco            # create & enter a session named "syco"
# ... run your python command inside ...
# detach (leave it running):   Ctrl-b  then  d
# reattach later:              tmux attach -t syco
# list sessions:               tmux ls
# stop the job:                reattach, then Ctrl-c
# kill the session:            tmux kill-session -t syco   (or type 'exit' inside)
```

IMPORTANT: tmux protects against your *browser/SSH* disconnecting. It does NOT survive the
Jupyter/on-demand *allocation* ending — so make sure your on-demand GPU session's walltime is
long enough for the run. If a run is cut off anyway, just re-run the same command: it resumes
from the last checkpoint.

## 3. Smoke test first (already passed for you)

```bash
python run_models.py --model Qwen/Qwen2.5-7B-Instruct --subset 25 --batch-size 32
```

## 4. Full run — all 6 models, inside tmux, no Slurm

```bash
tmux new -s syco
export HF_TOKEN=hf_xxxxxxxx
# export HF_HOME=$HOME/work/hf_cache
export HF_HOME=/project/lw29/sp3665/hf_cache

for M in \
  Qwen/Qwen2.5-7B           Qwen/Qwen2.5-7B-Instruct \
  mistralai/Mistral-7B-v0.3 mistralai/Mistral-7B-Instruct-v0.3 \
  meta-llama/Llama-3.1-8B   meta-llama/Llama-3.1-8B-Instruct ; do
    echo "=== $M ==="
    python run_models.py --model "$M" \
        --prompts prompts_out/prompts_all.jsonl \
        --batch-size 64
done
```

Then detach with `Ctrl-b d` and close your browser if you like. Reattach with
`tmux attach -t syco` to check the live progress bar.

Each model writes its own folder under `model_runs/<model-tag>/`, so they never overwrite.
The loop continues model-to-model; every run is resumable, so re-running the whole loop
after any interruption just continues where it stopped.

## 5. Live progress / ETA

The tqdm bar shows elapsed < remaining (ETA), batches/s, and the running sycophancy rate:
```
[Qwen2.5-7B-Instruct]:  46%|####5     | 180/390 [12:03<14:01, 0.25batch/s, done=5760, syc=0.24]
```
The ETA settles after ~10 batches (the first batch includes CUDA warmup, so early estimates
look slow). Read your real per-model time off the settled bar.

Rough expectation on an idle 80GB A100 with `--flash --batch-size 32`: on the order of
~45–90 min per model, so ~5–8 h for all six run back to back. Benchmark from the bar to be sure.

## 6. Key flags

- `--subset N`         N prompts per dataset (quick runs). Omit for the full set.
- `--batch-size`       32 (or 64 on your 80GB card).
- `--flash`            flash_attention_2 (safe, faster; needs flash-attn installed — drop the flag if not).
- `--max-new-tokens`   default 20.
- `--dtype`            bfloat16 (default).
- `--layers`           default 0 7 14 21 27.
- `--out-dir`          default model_runs/.
- `--checkpoint-every` flush every N batches (default 50) for resumability.

## 7. After each model — quick sanity check

```python
import numpy as np, pandas as pd
tag = "Qwen2-5-7B-Instruct"
hs = np.load(f"model_runs/{tag}/hidden_states.npz")
print({k: hs[k].shape for k in hs.files})          # 10 keys, each (n, 3584)
df = pd.read_parquet(f"model_runs/{tag}/responses.parquet")
print(df.shape)
print(df.groupby("dataset")[["sycophantic_gen","sycophantic_lp","pressured_parseable"]].mean())
```

For each BASE model, check `pressured_parseable` / the `pressured_unparseable_rate` in config.json:
if it's low, `sycophantic_gen` is trustworthy; if high, prefer `sycophantic_lp` for that model.
