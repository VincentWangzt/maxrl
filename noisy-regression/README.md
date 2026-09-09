# Fixed-pool noisy linear regression

Independent synthetic autoregressive SFT experiment. All Python execution is on
`cmu-L40-live:~/maxrl`; local work is editing, Git and Ruff only.

## Current experiment

The current launchers prepare **10,000,000 frozen training examples** and
**1,024 held-out evaluation examples**, then train on **GPU 1** for
**75,000 optimizer steps**, batch **128**, microbatch **128**. This consumes
**9,600,000 distinct examples**, or **0.96 pool passes**: no training example is
repeated in this run, and 400,000 examples remain unused.

Each example independently draws `w ~ N(0,I/2)`, 16 context inputs and one
query from `N(0,I_2)`. Outputs are `y = w·x + epsilon`, with independent context
and query noises sharing **sigma=0.01**. The prior scales with dimension to keep
the unconditional signal variance at one. Noise is added to outputs, not inputs.
Each prompt uses one common latent coefficient vector; different prompts have
different vectors. Continuous arrays, coefficients, noise realizations and
targets are generated once and saved. There is no test split.

There are **no fixed seed settings** in dataset generation or SFT. Dataset
splits and NumPy generators for shuffling and subset selection use fresh OS
entropy, and training does not reset Python, NumPy or Torch seeds. Each new
invocation therefore creates fresh random draws. Saved datasets stay fixed,
and checkpoint RNG states support resuming an existing run. Deterministic
Torch kernels remain enabled; that does not fix random initialization.

## Numerical vocabulary and prompt

Every scalar uses an inclusive 256-center grid on `[-3,3]`, midpoint ties toward
the larger index, then two base-16 digit IDs. Out-of-range values map to endpoint
bins; nonfinite scalars fail. The spacing is `6/255` (about 0.02353); there is no
exact zero center. A digit pair `(a,b)` decodes to `-3 + (16*a+b)*6/255`.
This spacing is larger than sigma=0.01. Clipping at +/-3 discards tail information;
the dataset metadata reports the actual clipping fraction for each array.

The **20-token vocabulary** is digits `0` through `F`, `[X]`, `[Y]`, `[PAD]`
and `[BOS]`, with IDs 0–19. The final query reuses the observation markers:

```text
[BOS]
[X] x_1 [Y] y_1
...
[X] x_16 [Y] y_16
[X] query_x [Y] query_y
```

Each `x` is two scalars (four digit tokens); each `y` is two digit tokens.
The prompt is **135 tokens**, ending with `[Y]`; its answer is two more tokens,
for **137 total**. The hidden `w`, unrounded arrays and noise values are never
included in the prompt. The vocabulary is saved as `codec.json`; there is no
text tokenizer or pretrained embedding.

Dataset **schema 3** records and validates the scalar range and prompt layout,
and rejects older pools explicitly. The preceding d=4, [-5,5], sigma=0.1 run
requires its original code revision `f2f6c2a`. Earlier 22-token experiments use
revision `1872344`. Historical datasets and checkpoints remain unchanged.

## Model and optimization

The scratch Qwen2 dimensions stay the same: hidden size 128, four layers,
four query heads/two KV heads, MLP size 512, context capacity 512, RoPE theta
1,000,000, RMSNorm epsilon 1e-6, gated SiLU, tied embeddings, full causal
attention and no dropout/sliding window/EOS. The 20-token vocabulary and
**987,776** trainable parameters are unchanged from the preceding run.

AdamW uses LR **1e-4**, betas `(0.9,0.95)`,
epsilon `1e-8`, weight decay `0.01`, **2,000 warmup updates then constant LR**, and
gradient clipping at 1.0. Update `s`, counted from 1, uses
`1e-4 * min(s/2000,1)`. The first update uses 5e-8 and update 2,000 reaches 1e-4.
Matrices decay; biases and RMSNorm scales do not.
Every parameter group is recorded. CUDA uses BF16 autocast with FP32 master
parameters/optimizer states, FP32 loss, and FP64 distribution statistics.

Only the final two answer tokens receive loss. The logits at positions 134
and 135 predict target positions 135 and 136 (zero-based), with both softmaxes
restricted to digit IDs 0–15. Loss is the batch mean of summed two-token NLLs.
Batch 128/microbatch 128 means one forward/backward pass per update.

## Server commands

Commit locally, push to GitHub, then pull into the server checkout. After sync,
run these commands on the server from `~/maxrl`:

```bash
bash noisy-regression/validate.sh
bash noisy-regression/prepare.sh
bash noisy-regression/sft.sh
# Optional final-checkpoint reevaluation:
bash noisy-regression/evaluate.sh
# CPU baselines on this same new evaluation pool:
bash noisy-regression/evaluate_bayesian.sh
bash noisy-regression/evaluate_ridge.sh
```

Launchers use explicit configuration blocks, the server `.venv`, and `.env`.
SFT enables online W&B logging in `noisy-regression-sft` and requires
`WANDB_API_KEY`. GPU launchers check that GPU 1 has no existing compute process.
Dataset generation and validation run on CPU. Output directories must be new.

### Quick learning-rate sweep

Run `bash noisy-regression/sweep_sft_lr.sh UNIQUE_SWEEP_NAME` on the server to
queue nine fresh SFT runs sequentially on **GPU 3**. Rates are
`1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4`; each uses
**10,000 steps** and **500 linear warmup steps, then constant LR**.
The launcher reuses `sft.sh` with explicit command-line overrides, keeping the
same 10M pool, batch/microbatch 128, architecture, optimizer, and evaluation
every 500 steps. Each run sees 1.28M distinct training examples (0.128 passes).
Initialization, training order and diagnostic subset remain independently
random per run, so this single-run sweep does not isolate seed variability.

Outputs are under `noisy-regression/checkpoints/UNIQUE_SWEEP_NAME/lrRATE/`,
with per-run logs in `logs/`, configuration in `config.txt`, and append-only
progress in `status.tsv`. All runs share the W&B group `UNIQUE_SWEEP_NAME`
in `noisy-regression-sft`. Each completed run produces the usual checkpoints,
metrics, plots and report. A failed run stops the queue and records its exit
code. The sweep reserves a GPU-specific lock and checks for compute processes
before each run. Use a detached server session to survive SSH disconnects.

## Evaluation and checkpoints

At step 0, every **500 updates**, and step 75,000, evaluation enumerates the
complete 256-answer distribution for all 1,024 held-out prompts. No completions
are sampled. One cached prefill plus 16 second-digit branches obtains these
probabilities. Checkpoints and exact per-prompt probabilities are retained.

W&B uses the existing 22 curated history keys in `eval`, `pass@k_exact`,
`train`, `diagnostics` and `timing`. MSE and NLL use
`eval/{mse,nll}/{clean,noisy}`; exact pass uses
`pass@k_exact/pass@{1,4,16,64,256}/{clean,noisy}`. MSE compares the exact
predictive mean with continuous targets; NLL and pass score quantized targets.
Detailed uncertainties remain in local artifacts. See [METRICS.md](METRICS.md).

Offline training NLL uses a random 1,024-example subset selected once per new
run. Its indices are stored in each checkpoint and recovered on resume, so
the diagnostic compares the same examples throughout a run. Held-out
evaluation always uses the complete evaluation pool. A final context-mismatch
control measures dependence on the context while preserving each query/target.

Bayesian and decoded-input ridge are separate baseline methods with the same
evaluation keys. The continuous Bayesian reference sees extra precision; ridge
uses quantized inputs and approximate Gaussian uncertainty. Both read sigma
from the new pool's metadata. Their run names include `d2_10m_xy_range3_sigma0p01`.

To resume, set `RESUME_CHECKPOINT` and a new `OUTPUT_DIR` in `sft.sh`, keeping
the complete training configuration unchanged. A checkpoint restores model,
optimizer, scheduler, shuffle/cursor, diagnostic subset, and Python/NumPy/
Torch/CUDA RNG state. The best pointer can reference an earlier run directory.
A resumed invocation starts a new W&B run with `resume_from` recorded.

## Artifacts

- Dataset: `noisy-regression/data/fixed_d2_n16_10m_xy_range3_sigma0p01/`, containing
  `train.npz`, `eval.npz`, `metadata.json` and `codec.json`. Archives are
  uncompressed to avoid compression overhead at this scale. All underlying
  continuous arrays, tokens, IDs and prompt hashes are retained. Metadata
  records file/content SHA-256, clipping and the train/eval overlap audit.
- Training: `noisy-regression/checkpoints/qwen2_1m_d2_10m_xy_range3_sft_75000_bs128_lr1e-4_warmup2000_sigma0p01/`,
  containing the manifest, dataset metadata, reference statistics, W&B run link,
  JSONL metrics, per-prompt evaluation archives, checkpoints, best pointer,
  final summary and plots/report generated after successful completion.
- Earlier runs are documented in [RESULTS.md](RESULTS.md) and
  [EVALUATION.md](EVALUATION.md). Their fixed seeds and older vocabulary describe
  those historical experiments, not the current launchers.

Focused server CPU checks cover fresh generation/frozen persistence, hashes,
split separation, the shared markers and answer-only loss, model parameter
count/causality, exact cached distributions, Gaussian references, W&B keys,
and full optimizer/diagnostic-subset recovery on resume. Test-only seeds make
numerical checks repeatable; experiment code does not set fixed seeds.
