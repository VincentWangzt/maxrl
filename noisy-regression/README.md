# Fixed-pool noisy linear regression

Independent synthetic autoregressive SFT experiment. All Python execution is on
`cmu-L40-live:~/maxrl`; local work is editing, Git and Ruff only.

## Current experiment

The current launchers prepare **10,000,000 frozen training examples** and
**1,024 held-out evaluation examples**, then train on **GPU 1** for
**150,000 optimizer steps**, batch **64**, microbatch **64**. This consumes
**9,600,000 distinct examples**, or **0.96 pool passes**: no training example is
repeated in this run, and 400,000 examples remain unused.

Each example independently draws `w ~ N(0,I/4)`, 16 context inputs and one
query from `N(0,I_4)`. Outputs are `y = w·x + epsilon`, with independent context
and query noises sharing **sigma=0.1**. Noise is added to outputs, not inputs.
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

Every scalar uses an inclusive 256-center grid on `[-5,5]`, midpoint ties toward
the larger index, then two base-16 digit IDs. Out-of-range values map to endpoint
bins; nonfinite scalars fail. The spacing is `10/255`; there is no exact zero
center. A digit pair `(a,b)` decodes to `-5 + (16*a+b)*10/255`.

The **20-token vocabulary** is digits `0` through `F`, `[X]`, `[Y]`, `[PAD]`
and `[BOS]`, with IDs 0–19. The final query reuses the observation markers:

```text
[BOS]
[X] x_1 [Y] y_1
...
[X] x_16 [Y] y_16
[X] query_x [Y] query_y
```

Each `x` is four scalars (eight digit tokens); each `y` is two digit tokens.
The prompt is **203 tokens**, ending with `[Y]`; its answer is two more tokens,
for **205 total**. The hidden `w`, unrounded arrays and noise values are never
included in the prompt. The vocabulary is saved as `codec.json`; there is no
text tokenizer or pretrained embedding.

Dataset **schema 2** uses this vocabulary and rejects older pools explicitly.
The earlier 22-token datasets/checkpoints require their original code revision
(the last revision before this change was `1872344`). They remain historical
artifacts and are not overwritten or silently converted.

## Model and optimization

The scratch Qwen2 dimensions stay the same: hidden size 128, four layers,
four query heads/two KV heads, MLP size 512, context capacity 512, RoPE theta
1,000,000, RMSNorm epsilon 1e-6, gated SiLU, tied embeddings, full causal
attention and no dropout/sliding window/EOS. Removing the two special tokens
reduces the trainable parameter count from 988,032 to **987,776**.

AdamW uses LR **1e-4** (one fifth of the former 5e-4), betas `(0.9,0.95)`,
epsilon `1e-8`, weight decay `0.01`, **200 warmup updates then constant LR**, and
gradient clipping at 1.0. Update `s`, counted from 1, uses
`1e-4 * min(s/200,1)`. Matrices decay; biases and RMSNorm scales do not.
Every parameter group is recorded. CUDA uses BF16 autocast with FP32 master
parameters/optimizer states, FP32 loss, and FP64 distribution statistics.

Only the final two answer tokens receive loss. The logits at positions 202
and 203 predict target positions 203 and 204 (zero-based), with both softmaxes
restricted to digit IDs 0–15. Loss is the batch mean of summed two-token NLLs.
Batch 64/microbatch 64 means one forward/backward pass per update.

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

## Evaluation and checkpoints

At step 0, every **500 updates**, and step 150,000, evaluation enumerates the
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
from the new pool's metadata. Their new run names include `10m_xy_sigma0p1`.

To resume, set `RESUME_CHECKPOINT` and a new `OUTPUT_DIR` in `sft.sh`, keeping
the complete training configuration unchanged. A checkpoint restores model,
optimizer, scheduler, shuffle/cursor, diagnostic subset, and Python/NumPy/
Torch/CUDA RNG state. The best pointer can reference an earlier run directory.
A resumed invocation starts a new W&B run with `resume_from` recorded.

## Artifacts

- Dataset: `noisy-regression/data/fixed_d4_n16_10m_xy_sigma0p1/`, containing
  `train.npz`, `eval.npz`, `metadata.json` and `codec.json`. Archives are
  uncompressed to avoid compression overhead at this scale. All underlying
  continuous arrays, tokens, IDs and prompt hashes are retained. Metadata
  records file/content SHA-256, clipping and the train/eval overlap audit.
- Training: `noisy-regression/checkpoints/qwen2_1m_fixed10m_xy_sft_150000_bs64_lr1e-4_sigma0p1/`,
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
