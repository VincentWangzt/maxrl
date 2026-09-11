# Reading the noisy-regression metrics

The dashboard should answer three questions: is the model learning the underlying
regression, does it assign probability to clean/noisy answers, and how often would
samples match each target? All dashboard scores use the exact distribution over
the full held-out pool. Start with clean MSE, noisy NLL, and exact pass@k.
Lower MSE/NLL is better; higher pass@k is better.

## What are the three targets?

These are three versions of the same example's target, not three model outputs.

| Stored name | Meaning | What an error against it measures |
|---|---|---|
| `continuous_noiseless_signal` | `s = w · x_query`, before noise or rounding | Recovery of the underlying regression function |
| `continuous_noisy_outcome` | `y = s + epsilon`, before rounding | Prediction error against the particular noisy observation |
| `decoded_target_grid_value` | `decode(encode(y))`, the center represented by the two target tokens | Error against the numerical answer the model actually trains on |

For example, a signal of 1.20 plus noise of 0.001 gives an outcome of 1.201.
The codec rounds that outcome to approximately 1.207843. Its grid has 256 centers
from -4 to 4, spaced by `8/255 ≈ 0.031373`; values beyond the range map to the
endpoints. “Continuous” means the original numerical value before this codec.
“Noiseless” removes query noise, but the predictor still has noisy context and
uncertainty about the unknown coefficients.

The model produces a distribution over 256 two-token answers. Decoding each
answer yields one grid center. Its **exact predictive mean** is the weighted
average of all 256 centers using their probabilities; that average can lie
between grid centers. Its **sampled predictive mean** averages the decoded values
of 256 random completions. Neither is a single sampled answer or the most likely
answer. Signal MSE squares the difference between the mean prediction and `s`,
then averages across examples. It does not average individual samples' squared
errors; that would additionally penalize the predictive distribution's spread.

For a predictor independent of fresh query noise,
`E[(prediction - y)^2] = E[(prediction - s)^2] + sigma^2`.
The current pool uses sigma=0.1, so the expected added error is 0.01.
The identity is an expectation, not an exact
difference on a finite frozen pool. Quantization/clipping further changes the
decoded-target error.

**Dashboard choice:** `eval/mse/clean` scores the continuous signal;
`eval/mse/noisy` scores the continuous noisy outcome. Both use the exact
predictive mean. Decoded-target errors, MAE and bias remain in evaluation
artifacts, without additional dashboard curves.

## Likelihood and pass@k answer different questions

- **Answer NLL:** `-log p(target_tokens)`, averaged over examples, in nats per
  complete two-token answer. `eval/nll/clean` scores `encode(s)` and
  `eval/nll/noisy` scores `encode(y)`. Neither is a continuous Gaussian density.
  Answer log-likelihood is exactly NLL's negative and adds no information.
- **Exact pass@k:** the average of `1 - (1 - p(target_tokens))^k`. All 256 answer
  probabilities are available, so this expected sampling success is computable
  without drawing completions. “Exact” refers to the calculation, not perfect
  prediction of the regression signal. Clean pass scores `encode(s)`; noisy
  pass scores `encode(y)`. Each prompt's success probability is transformed
  before averaging across prompts.
- **Sampled pass@k:** an estimate from 256 actual completions per example. For
  `c` matching completions it is `1 - C(256-c,k)/C(256,k)`. This is a sampling
  check retained in historical artifacts, not a current evaluation metric. At k=256 it records
  whether at least one completion matched the stored target.

High pass@256 can coexist with a poor mean prediction: many guesses can cover
the noisy target. Conversely, a good predictor cannot know fresh query noise.
Signal MSE and NLL are therefore both useful. The dashboard keeps exact
k=1,4,16,64,256 for both targets; artifacts retain all nine k values.

Clean and noisy scores evaluate the **same predictive distribution**. Changing
the scoring target does not remove noise from context examples, change training
labels, or deconvolve query noise out of the model's distribution. The clean
NLL/pass curves show how much probability that distribution assigns to the
rounded signal. Even a good mean estimate can have low clean pass@1 if its
distribution appropriately includes observation noise.

## What do the references mean?

These are fixed analytical predictors evaluated on the same held-out examples.
They are not additional trained networks or ground-truth labels. Their scores
are fixed because the dataset does not change. Gaussian reference probabilities
are integrated over codec bins, including the infinite tails of endpoint bins,
so their NLL and pass@k score the same discrete answers as the model.

| Reference in the artifacts | Information used | Where it is recorded |
|---|---|---|
| `uniform_256` | No inputs; equal probability for every answer | Artifact only: trivial chance baseline |
| `query_only_continuous_optimistic` | Original query input, coefficient prior and noise level; ignores all context | Artifact only: precision comparison |
| `query_only_decoded_plugin_approximation` | Quantized query input, coefficient prior and noise level; ignores all context | Offline no-context baseline |
| `ridge_decoded_gaussian_approximation` | Quantized context inputs/outcomes and query, prior and noise level | Separate `ridge` run using the shared evaluation keys |
| `bayesian_continuous_optimistic` | Original continuous context inputs/outcomes and query, prior and noise level | Separate `bayesian` run using the shared evaluation keys |

The query-only predictor is centered at zero; its spread depends on the query
norm and noise level. Uniform and both query-only variants therefore have the
same mean prediction and signal MSE, even though their NLL/pass@k differ.

Bayesian regression infers the coefficients from the 16 context examples under
the known Gaussian prior. The continuous version has extra input precision.
The ridge version substitutes decoded grid centers into that Gaussian model;
it does not integrate over every possible continuous value represented by the
tokens. It is an approximation to the posterior given quantized observations,
not the exact Bayes solution for the model's input. Neither reference uses the
hidden coefficients or the query outcome to make its prediction. Both use the
known generative prior and sigma, which the learned model must acquire through
training. Their reported signal MSE uses their grid-distribution mean, just as
the model's does. For clean NLL/pass, these baselines also score their noisy
outcome distribution at the clean target; they do not switch to a narrower
posterior distribution of the noiseless signal.

For context, the completed `qwen2_1m_fixed100k_sft_10000_bs64x1_sigma0p1` run at
step 10,000 has full-pool signal MSE **1.084363**, versus **1.067846** for the
zero-mean query-only reference, **0.004139** for quantized ridge, and **0.003734**
for continuous Bayes. On this pool, the mean prediction has not beaten the
no-context baseline. Query noise and the continuous/quantized reference gap do
not explain that large error. This observation concerns one run and its reused
evaluation pool, not an independent test or a proof of architecture limits.

## Why were there 535 reference metrics?

The old logger recursively turned every scalar into a W&B history key.
Each reference had 21 measurements:

- NLL, its redundant negative (log-likelihood), and entropy: 3.
- Exact pass@k at nine k values: 9.
- MSE, MAE, and bias against each of the three targets: 9.

Each measurement expanded into five keys: mean, prompt SE, lower and upper
normal-95% bounds, and prompt count. Another prompt count and the maximum
probability-normalization error were logged at the predictor level.
Thus **5 predictors × (21 × 5 + 2) = 535 keys**.

The completed run's entire history had **827 experiment-defined scalar keys**:

| Old top level | Keys |
|---|---:|
| `reference` | 535 |
| `eval` | 276 |
| `train_eval` | 10 |
| `train` | 3 |
| `trainer` | 3 |

The evaluation group repeated the same 107-key distribution report, plus 159
generation keys and 10 final shuffled-context likelihood keys. Generation
included sampled pass, exact pass on the same subset, their difference,
uncertainty/counts for every k, sampled-mean MSE, and scalar sampling settings.
Most keys were supporting statistics or metadata, not independent outcomes.

## Current dashboard: 24 history keys for training

Logging now selects named metrics explicitly. Adding a number to an evaluation
artifact can no longer silently create another panel. Each method has its own
run and uses the same evaluation names. No baseline curves are injected into
a learned method's run, and there is no `reference/` metric namespace.

| Top level | Curves/fields | Count |
|---|---|---:|
| `eval` | `{mse,nll}/{clean,noisy}` | 4 |
| `pass@k_exact` | `pass@{1,4,16,64,256}/{clean,noisy}` | 10 |
| `train` | `answer_nll`, `learning_rate`, `gradient_norm` | 3 |
| `train_batch` | `mse/{clean,noisy}` on the just-optimized batch | 2 |
| `diagnostics` | `predictive_entropy_nats`, final-only `context_shuffle_nll_increase` | 2 |
| `timing` | `elapsed_seconds`, `optimizer_step_seconds`, `evaluation_seconds` | 3 |

The shuffled-context diagnostic is shuffled-context NLL minus normal evaluation
NLL. A positive value shows that replacing the examples hurts prediction; it
does not demonstrate correct regression inference. Entropy measures predictive
spread and has no universal better direction.

There are no dashboard SEs, normal-95% bounds, prompt counts, sampled scores,
training-batch NLL/pass scores, grid-target errors, repeated sampling settings,
or progress section. Invalid distributions still fail validation in the
evaluator. Fixed dataset sizes and method settings live in run config. W&B's
native step is the optimizer step; no duplicate step metric is logged.

Every `eval` MSE/NLL/pass metric uses all 1,024 held-out examples at every
evaluation, including step 0 and the final step. At each nonzero evaluation,
the post-update model also enumerates all 256 answers on the just-optimized
effective batch (1,024 examples canonically; 512 or 2,048 in the batch sweep).
Its clean/noisy MSE is logged under `train_batch`; the batch
IDs and complete distribution summary stay in the JSONL artifact. This measures
immediate training fit, not a stable population estimate, so it is expected to
be noisier and more optimistic than held-out MSE. Evaluation NPZ files contain
only full-pool IDs and exact log probabilities. No completions are sampled.

`train/gradient_norm` is the global L2 norm across all model parameter gradients
after backward passes over the effective batch, before any optional clipping.
`max_grad_norm` is recorded in the training configuration: `null` disables
clipping (the default); a finite positive value caps the norm for the update.
The norm is logged with loss/LR at step 1 and then every `log_interval` steps
(10 in the launcher); a nonfinite norm fails the update explicitly in either mode.

`elapsed_seconds` measures total wall time since the invocation started, including
restored elapsed time on resume. `optimizer_step_seconds` measures the logged
update's optimization work, excluding subsequent logging/evaluation/checkpoint
writing. `evaluation_seconds` measures held-out and training-batch evaluation,
the evaluation NPZ write, and the final context control when present; it excludes
model checkpoint writing. Baseline evaluation
time measures its analytical distribution and metric calculation, excluding
dataset loading and artifact writing. These are wall times, not GPU kernel times.

Best NLL/step and checkpoint paths are recorded once in the run summary.
The 24-key count excludes W&B's own system metrics and these summary fields.
Checkpoint selection still uses the lowest noisy evaluation NLL. New evaluation
artifacts add `clean_answer_nll` and `clean_exact_pass` alongside the existing
noisy fields; previously saved artifacts are not rewritten.

## One-time Bayesian and ridge runs

After the usual Git synchronization, run either CPU-only launcher on the server:

```bash
bash noisy-regression/evaluate_bayesian.sh
bash noisy-regression/evaluate_ridge.sh
```

Each creates a separate run in `noisy-regression-sft` and logs **once at step 0**.
Current run names are `bayesian_continuous_d2_n64_10m_sep_eoo_range4_sigma0p1` and
`ridge_quantized_d2_n64_10m_sep_eoo_range4_sigma0p1`, using the 10M-pool evaluation split.
Each records the same 14 evaluation/pass scores, entropy, and two timing fields:
17 history keys total. No completions are sampled and no model is trained.
The shared keys allow comparing methods in the same panel or run-summary table;
a step-zero baseline is a single point, not a repeated horizontal history curve.

Paths, project/run names, and `USE_WANDB` are explicit in each launcher. Both
require a new output directory and refuse overwrite. The shared implementation
is `noisy_regression.evaluate_baseline`; it records the dataset fingerprint,
method/input precision, metrics, and exact probabilities in that new directory.

This schema applies to newly started logger instances. Existing W&B histories
and workspace panels are not rewritten by a code update; the completed run
retains its old keys. New runs use the new groups. This change does not replay,
rename, or upload any previous run's artifacts.
# Exact population training objectives

For each prompt, `p(a,b)=p(a|prompt)p(b|prompt,a)` covers all 256 digit pairs.
Let `v(a,b)=-4+(16a+b)*8/255` and let `y` be the decoded **observed target
digits**. Training averages one loss per prompt, without dividing by two tokens.
The following objectives replace both answer-digit and EOS SFT losses:

- **RLOO:** `E_p[(v-y)^2]`, the exact population-limit RLOO gradient. The expected
  score of the population mean baseline vanishes; no finite-group correction is
  applied to the 256 enumerated actions.
- **GRPO:** `-sum stopgrad(p*A)*log(p)`, with reward `r=-(v-y)^2` and
  `A=(r-E_p[r])/(sqrt(Var_p[r])+epsilon)`. All moments are probability weighted,
  and the score weights are detached. Default epsilon is `1e-8`. Its logged loss
  is a score-function surrogate, not a scalar objective comparable across methods.
- **MaxRL:** Gaussian reward `s=exp(-(v-y)^2/(2*tau^2))`, with `Z=E_p[s]`.
  Degree `d` minimizes `sum_{k=1}^d (1-Z)^k/k`; `inf` minimizes `-log(Z)`.
  The finite-degree gradient equals the log-objective gradient multiplied by
  `1-(1-Z)^d`. Log-space reward normalization prevents underflow from clipping
  the infinite-degree gradient. The logged loss is the actual scalar objective,
  and the surrogate gives its exact gradient. `tau` is independent of data sigma.

Optimization events contain a `population` dictionary, mirrored as
`train/population/*` in W&B. Evaluation dictionaries contain the same metrics
with mean/SE, mirrored as `eval/population/*`:

| Metric | Definition |
| --- | --- |
| `loss` | RLOO objective, GRPO surrogate, or MaxRL objective as above |
| `expected_mse` | `E_p[(v-y)^2]`, exact single-sample MSE |
| `predictive_mean_mse` | `(E_p[v]-y)^2`, against decoded observed target |
| `predictive_variance` | `Var_p[v]`; expected MSE = mean MSE + this variance |
| `reward_mean`, `reward_std` | Population moments of negative squared error |
| `entropy_nats` | Joint two-digit entropy |
| `answer_nll` | Nats at the observed digit pair |
| `maxrl_expected_reward` | `Z`, MaxRL only |
| `maxrl_log_expected_reward` | `log Z`, remains usable when `Z` underflows |
| `maxrl_gradient_scale` | `1-(1-Z)^d`, or 1 for infinite degree |

Existing clean/noisy continuous predictive-mean MSE, quantized NLL, pass@k,
entropy, learning rate and gradient norm remain available for comparison with
SFT. Following nanochat's scalar evaluation, `eval/target_variance/{clean,noisy}`
and `eval/mse_over_target_variance/{clean,noisy}` use the population variance of
the held-out targets (ddof=0), not model predictive variance or known noise
variance. A constant target pool records a null ratio. Checkpoint selection
continues to use held-out noisy answer NLL; this criterion differs from the RL
objectives and should be considered when comparing best checkpoints.
