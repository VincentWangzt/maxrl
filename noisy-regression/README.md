# Canonical noisy linear regression

All Python execution runs on `cmu-L40-live:~/maxrl`. Commit locally, push to
GitHub, then pull on the server before preparing data, validating or training.

## Configuration

| Setting | Canonical value |
| --- | --- |
| Frozen training / held-out evaluation pool | 10,000,000 / 1,024 examples |
| Dimension / context observations | 2 / 64 |
| Context and query noise standard deviation | 0.001 |
| Numerical codec | 256 inclusive centers on [-4,4]; two base-16 digits per scalar |
| Model | Scratch Qwen2, 4 layers, hidden 128, MLP 512; 988,288 parameters |
| Effective batch / microbatch | 1,024 / 128 (8 accumulation passes) |
| Optimizer steps | 20,000 |
| Learning rate | 1e-4 |
| Warmup | 200 updates, from 10% to 100% of the learning rate |
| After warmup | Constant learning rate; no LR decay |
| AdamW | betas (0.9,0.95), epsilon 1e-8, weight decay 0.01 |
| Gradient clipping | Disabled |
| Precision | BF16 autocast; FP32 parameters and optimizer state |
| Evaluation and checkpoints | Step 0, every 500 steps, and final step 20,000 |

One-based update `s` for `1 <= s <= 200` uses
`learning_rate * (0.1 + 0.9 * (s - 1) / 199)`. Update 1 is exactly 10% and
update 200 reaches the peak. Updates 201 through 20,000 keep that peak.
`warmup_start_factor` is independent of the optional cosine decay floor, so
changing the peak LR preserves the 10% start. AdamW still decays matrices;
biases and RMSNorm scales do not decay.

Each problem draws `w ~ N(0,I/2)` and 64 context inputs plus one query from
`N(0,I_2)`. Outputs use `y = w·x + epsilon`, with independent context and query
noise. A prompt shares one coefficient vector; separate problems have separate
vectors. Continuous arrays, coefficients, noise, targets, tokens, IDs and hashes
are frozen in the dataset. There is no test split.

## Prompt and model

```text
[BOS]
[X] x_1,1 [SEP] x_1,2 [Y] y_1 [EOO]
...
[X] x_64,1 [SEP] x_64,2 [Y] y_64 [EOO]
[QUERY] [X] query_x_1 [SEP] query_x_2 [Y] answer [EOS]
```

Every displayed scalar expands to two digit tokens. Each observation has
10 tokens. The prompt has **649 tokens**, ending in `[Y]`; the completion is
two answer digits and `[EOS]`, giving **652 total tokens**. The 24-token
vocabulary is digits 0–F, `[X]`, `[SEP]`, `[Y]`, `[EOO]`, `[QUERY]`, `[PAD]`,
`[BOS]`, `[EOS]`, with IDs 0–23. Latents and unrounded numbers never enter the
model input. Only the completion receives loss: two digit-restricted NLLs and
a full-vocabulary EOS NLL. Numerical answer metrics exclude the EOS loss.

The codec rounds to the nearest center, breaks midpoint ties toward the
larger index, clips tails to endpoints and rejects nonfinite inputs.
A digit pair `(a,b)` decodes to `-4 + (16*a+b)*8/255`. The spacing, approximately
0.03137, is substantially larger than sigma=0.001; tokenization hides most
noise realizations. The wider range reduces clipping but makes bins coarser
than the preceding [-3,3] codec. Metadata records actual clipping fractions.

Qwen2 uses four query heads, two KV heads, capacity 1,024, RoPE theta 1,000,000,
RMSNorm epsilon 1e-6, gated SiLU, tied embeddings and full causal attention.
Dropout and sliding windows are disabled. Dataset generation, initialization
and shuffle use fresh randomness without fixed seeds. Checkpoints retain RNG
and shuffle state for resumption.

## Launching

From the server checkout, after Git synchronization:

```bash
bash noisy-regression/validate.sh
bash noisy-regression/prepare.sh
# Canonical run alone:
bash noisy-regression/sft.sh --gpu-id 2
# OR canonical run plus four simultaneous sweep runs:
nohup bash noisy-regression/sweep_sft.sh canonical_20260911 > noisy-regression/launch.log 2>&1 < /dev/null &
```

The sweep calls the same `sft.sh` for all five runs:

| GPU | Batch | LR | Warmup starting LR | Presentations / pool passes |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1,024 | 1e-4 | 1e-5 | 20,480,000 / 2.048 |
| 6 | 512 | 1e-4 | 1e-5 | 10,240,000 / 1.024 |
| 7 | 2,048 | 1e-4 | 1e-5 | 40,960,000 / 4.096 |
| 8 | 1,024 | 5e-5 | 5e-6 | 20,480,000 / 2.048 |
| 9 | 1,024 | 2e-4 | 2e-5 | 20,480,000 / 2.048 |

Every run uses the same frozen pools and 20,000 updates. The batch comparison
changes both gradient batch size and total presentations. Independent
initialization and shuffling also contribute variation; this is a single-run
sweep, not a replicated estimate of each setting's effect.

Launchers use the server `.venv` and `.env`, require `WANDB_API_KEY`, and log
online to `noisy-regression-sft`. They check that selected GPUs are free and
refuse existing outputs. The sweep preflights all five devices, records PIDs
and logs, and records exit codes when waiting for each child. Preparation and
validation are CPU-only. Each dataset/output directory must be new. The
standalone canonical command and the sweep's GPU 2 job target the same output;
launch one or the other.

## Evaluation and artifacts

`config.sh` holds the shared dataset and run identity. Canonical data lives in
`noisy-regression/data/fixed_d2_n64_10m_sep_eoo_range4_sigma0p001/`:
uncompressed `train.npz`, `eval.npz`, `metadata.json` and `codec.json`.
Schema **6** rejects legacy pools. Metadata includes SHA-256 file/content
hashes, clipping and a train/eval prompt-overlap audit. Loading verifies hashes
and codec settings before training.

Outputs live under `noisy-regression/checkpoints/` with names
`canonical_d2_n64_10m_sep_eoo_range4_sigma0p001_bsBATCH_lrRATE/`. They include
configuration, dataset fingerprints, Git commit, W&B link, metrics, per-prompt
distributions, checkpoints and a final report. Sweep logs and its run table
live in `noisy-regression/logs/SWEEP_NAME/`.

Evaluation enumerates all 256 two-digit answers on all 1,024 held-out prompts
using a cached prefill and 16 second-digit branches. It also evaluates the
just-optimized effective training batch at nonzero evaluation steps. A final
context-mismatch control checks context dependence. The held-out pool is reused
for checkpoint selection. See [METRICS.md](METRICS.md) for definitions.

After training, `evaluate.sh` reevaluates the canonical final checkpoint on
GPU 2 (which must be free). `evaluate_bayesian.sh` and `evaluate_ridge.sh` run
CPU baselines on the same evaluation pool. The continuous Bayesian baseline
sees extra precision; decoded-input ridge uses an approximate uncertainty model.

For recovery, set `RESUME_CHECKPOINT` and a new `OUTPUT_DIR` in `sft.sh`.
Settings must match the checkpoint, except that the horizon may be increased.
RNG, optimizer and shuffle state are restored. Legacy datasets, checkpoints,
sweep scripts and reports have been retired; historical source and reports
remain available through Git history.
