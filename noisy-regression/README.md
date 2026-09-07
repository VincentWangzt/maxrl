# Fixed-pool noisy linear regression

Independent synthetic autoregressive SFT experiment, with no dependency on the
maze tokenizer, rewards, or datasets. All Python execution is on
`cmu-L40-live:~/maxrl`; local work is editing, Git and Ruff only.

## Confirmed first experiment

The user confirmed **GPU 1** and **100,000 frozen training examples**. There are
1,024 held-out evaluation examples and no test split. Each example draws a new
`w ~ N(0,I/4)`, 16 context inputs and one query from `N(0,I_4)`, and independent
context/query noise with standard deviation 0.5. Continuous outcomes are formed
before quantization. All latents, continuous arrays, noises and targets are saved
once, including the noisy query outcome.

Every scalar uses the inclusive 256-center grid on `[-5,5]`, midpoint ties toward
the larger index, then two base-16 digit IDs. The visible sequence is 205 tokens;
the prompt is its first 203 tokens. Nonfinite scalars and overlength input fail.
The codec uses midpoint search, the floating-point stable equivalent of the
specified clipped rounding formula, so exact represented midpoint ties and their
immediate neighbors have unambiguous results. There is no exact zero center.

The 22-token vocabulary is saved as `codec.json`. Numeric encoding/decoding is
explicit (`noisy_regression.codec`), with no text tokenizer or pretrained
weights. The model is scratch Qwen2: hidden size 128, four layers, four query
heads/two KV heads, MLP size 512, context capacity 512, RoPE theta 1,000,000,
RMSNorm epsilon 1e-6, gated SiLU, tied embeddings, full causal attention, no
dropout/sliding window/EOS. Architecture values are explicitly passed from the
configuration block in `sft.sh`. The expected parameter count is **988,032**;
the run computes the actual count and records it in `manifest.json`.

Training uses exactly 10,000 updates, effective batch 64, microbatch 16, four
accumulation rounds, and 640,000 presentations (6.4 pool passes). Shuffling spans
epoch boundaries without dropping or regenerating examples. AdamW uses LR
`5e-4`, betas `(0.9,0.95)`, epsilon `1e-8`, weight decay `0.01`, 200 warmup
updates then constant LR, and gradient norm clipping at 1.0. Update `s`, counted
from 1, uses `5e-4 * min(s/200,1)`. Matrices, including the tied embedding, decay;
biases and RMSNorm scales do not. Every parameter name in each group is recorded.
CUDA forward uses BF16 autocast with FP32 master parameters and optimizer states;
loss/log-softmax are FP32 and probability/statistical calculations are FP64.
Deterministic algorithms are enabled; cross-version/device bitwise identity is
not promised. Resume must use the same settings and dataset.

## Server commands

Commit locally, push to GitHub, then pull those commits into the server checkout.
Do not copy source files to the server. The following commands run on the server
from `~/maxrl`, after Git synchronization:

```bash
bash noisy-regression/validate.sh
bash noisy-regression/prepare.sh
bash noisy-regression/sft.sh
# Optional independent reevaluation of the final checkpoint after training:
bash noisy-regression/evaluate.sh
```

Each launcher has its own paths, experiment values and environment block; there
is no shared `config.sh` and no ambient experiment-setting overrides. GPU work
fails if the explicitly selected GPU has a compute process. Launchers source
the repository `.env` using the existing convention; this experiment writes
local JSON/NPZ logs and does not require an external tracking service. Never
print credentials. The existing server `.venv` must provide PyTorch,
Transformers with Qwen2/`DynamicCache.batch_repeat_interleave`, NumPy, SciPy,
Matplotlib and pytest. Exact installed versions are captured with each run.

Data preparation refuses an existing output directory. Training and standalone
evaluation also require new output directories. To resume, set
`RESUME_CHECKPOINT` and a new `OUTPUT_DIR` in the top block of `sft.sh`, leaving
the complete experiment configuration unchanged. Resume restores optimizer,
scheduler, Python/NumPy/Torch/CUDA RNGs, current shuffle, cursor, epoch counter,
presentation count and best checkpoint. Prior checkpoints remain where they
were saved; the best pointer may reference the original run directory. A
checkpoint directory is published only after all its files are written.

## Evaluation and interpretation

At step 0, every 500 updates, and step 10,000, likelihood and exact pass@k use all
1,024 held-out examples. Comparable training NLL uses a fixed 1,024-example
training subset. The loss is the mean of **summed two-token answer NLLs**, with
the logits at positions 202 and 203 predicting target positions 203 and 204
(zero-based). Both softmaxes contain only digit IDs 0–15; Hugging Face's
unrestricted, internally shifted loss is not used.

`exact_pass[k]` is the prompt average of `1-(1-p_target)^k`, evaluated stably.
`generation.generative_pass[k]` is the combinatorial estimator from 256 actual
independent digit-pair completions per prompt, for
`k = 1,2,4,8,16,32,64,128,256`. Periodic generation uses a fixed 128-example
subset; final generation uses all 1,024 examples. Per-prompt prefill is cached
and branched over 16 first digits to get all conditional second-digit
distributions. Sampling then draws a first digit followed by a second digit
conditioned on that sampled first digit, from the cached probabilities on CPU.
Temperature is 1, without truncation or beams; output length is exactly 2.
`eval_batch_size` controls GPU prompt batches; `generation_batch_size` controls
CPU prompt batches when sampling those distributions. The complete samples,
IDs, success counts and joint log-probability tables are saved.

Exact and generated metrics are compared on the same selected prompts. Prompt
standard errors and approximate normal intervals use examples as units;
conditional Monte Carlo standard deviations integrate the estimator over each
prompt's `Binomial(256,p_target)` success count. They do not count completions as
additional regression examples or capture training-seed variation. A fixed
sampling seed plus checkpoint step is used, independently of training RNGs.
Changing generation batch size changes the sample stream assignment, not the
distribution. Fixed subset indices come from one separate PCG64 stream: a
training permutation followed by an evaluation permutation.

Reference metrics include uniform 256-way predictions, query-only continuous
prior predictions, their decoded-query plug-in approximation, continuous-data
Bayesian regression, and decoded-data ridge/Gaussian regression. Cholesky solves
avoid explicit inverses. Gaussian probabilities integrate between grid
midpoints, with infinite endpoint tails and stable log-CDF differences. The
continuous references see extra precision and are labeled optimistic; the
decoded variants are approximations to conditioning on quantized observations.

Reports include entropy, normalization error, clipping fractions, predictive
mean errors separately against continuous noisy outcomes, decoded target
centers and continuous noiseless signals, and a final mismatched-context NLL
control that preserves each query/target. A rise under the control indicates
context dependence, not Bayes-optimal inference. Held-out data are reused for
selection and are never passed to the optimizer. One seed, a frozen noisy pool,
and evaluation-based selection limit generalization claims. No target pass@1
or architecture adequacy claim is assumed.

## Artifacts and seeds

- Dataset: `noisy-regression/data/fixed_d4_n16_100k/{train,eval}.npz`,
  `metadata.json`, `codec.json`. NPZ arrays include continuous inputs/outcomes,
  coefficients, noise, noiseless query signal, token sequences, stable IDs and
  prompt hashes. Metadata includes clipping counts, exact configuration,
  content SHA-256, file SHA-256 and the train/eval prompt-overlap audit. File
  hashes verify the stored archive; content hashes identify deterministic
  arrays independently of archive container metadata.
- Run: `noisy-regression/checkpoints/qwen2_1m_fixed100k_sft_10000/` with
  `manifest.json`, `references.json`, `metrics.jsonl`, `evaluation-*.npz`,
  `checkpoint-00000` through `checkpoint-10000`, `best_checkpoint.json`,
  `summary.json`, `report.md`, `learning_curves.png` and `pass_at_k.png`.
  Step 0 is retained and eligible for best-checkpoint selection.
- Seeds: training data 1729; evaluation data 2718; model/global 3141;
  training order 1618; fixed subsets 5772; completion sampling 8119 + step.

The SFT launcher creates the report after successful completion. It can also be
rendered while the run is in progress on the server:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/noisy-regression:$PWD" \
  .venv/bin/python -m noisy_regression.report \
  --run noisy-regression/checkpoints/qwen2_1m_fixed100k_sft_10000
```

Focused CPU validation covers codec endpoints/ties/clipping/round trips,
reproducibility/fingerprints/split separation, frozen examples across epochs,
sequence layout, answer-only shifting and masking, actual model parameter
count/forward/backward/causality, cached distribution agreement, normalization,
sampling/estimator correctness, Gaussian references, checkpoint resume, and a
two-update end-to-end run. Run only that module, not the repository's full suite.
