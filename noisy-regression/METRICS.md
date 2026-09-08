# Reading the noisy-regression metrics

The dashboard should answer three questions: is the model learning the underlying
regression, does it assign probability to the observed answers, and how often do
samples match the target? Start with signal MSE, answer NLL, and exact pass@k.
Lower MSE/NLL is better; higher pass@k is better.

## What are the three targets?

These are three versions of the same example's target, not three model outputs.

| Stored name | Meaning | What an error against it measures |
|---|---|---|
| `continuous_noiseless_signal` | `s = w · x_query`, before noise or rounding | Recovery of the underlying regression function |
| `continuous_noisy_outcome` | `y = s + epsilon`, before rounding | Prediction error against the particular noisy observation |
| `decoded_target_grid_value` | `decode(encode(y))`, the center represented by the two target tokens | Error against the numerical answer the model actually trains on |

For example, a signal of 1.20 plus noise of 0.10 gives an outcome of 1.30.
The codec rounds that outcome to approximately 1.313725. Its grid has 256 centers
from -5 to 5, spaced by `10/255 ≈ 0.039216`; values beyond the range map to the
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
The current pool uses sigma=0.1, so the expected added error is 0.01; the original
pool used sigma=0.5, giving 0.25. The identity is an expectation, not an exact
difference on a finite frozen pool. Quantization/clipping further changes the
decoded-target error.

**Dashboard choice:** keep signal MSE. The noisy-outcome and decoded-target
MSE/MAE/bias families help audit noise and quantization, but do not need separate
learning curves. They remain in the saved evaluation JSON.

## Likelihood and pass@k answer different questions

- **Answer NLL:** `-log p(target_tokens)`, averaged over examples, in nats per
  complete two-token answer. It scores the noisy, quantized target distribution.
  Answer log-likelihood is exactly its negative and adds no information.
- **Exact pass@k:** the average of `1 - (1 - p(target_tokens))^k`. All 256 answer
  probabilities are available, so this expected sampling success is computable
  without drawing completions. “Exact” refers to the calculation, not perfect
  prediction of the regression signal.
- **Sampled pass@k:** an estimate from 256 actual completions per example. For
  `c` matching completions it is `1 - C(256-c,k)/C(256,k)`. This is a sampling
  check, not another independent measure of learning. At k=256 it records
  whether at least one completion matched the stored target.

High pass@256 can coexist with a poor mean prediction: many guesses can cover
the noisy target. Conversely, a good predictor cannot know fresh query noise.
Signal MSE and NLL are therefore both useful. Sampled pass curves are secondary;
the dashboard keeps k=1,16,256, and the artifacts retain all nine k values.

## What do the references mean?

These are fixed analytical predictors evaluated on the same held-out examples.
They are not additional trained networks or ground-truth labels. Their curves
are constant because the dataset does not change. Gaussian reference probabilities
are integrated over codec bins, including the infinite tails of endpoint bins,
so their NLL and pass@k score the same discrete answers as the model.

| Reference in the artifacts | Information used | Role in the dashboard |
|---|---|---|
| `uniform_256` | No inputs; equal probability for every answer | Artifact only: trivial chance baseline |
| `query_only_continuous_optimistic` | Original query input, coefficient prior and noise level; ignores all context | Artifact only: precision comparison |
| `query_only_decoded_plugin_approximation` | Quantized query input, coefficient prior and noise level; ignores all context | `query_only`: measures what is possible without learning from the examples |
| `ridge_decoded_gaussian_approximation` | Quantized context inputs/outcomes and query, prior and noise level | `ridge_quantized`: regression baseline with the model's input precision |
| `bayesian_continuous_optimistic` | Original continuous context inputs/outcomes and query, prior and noise level | `bayes_continuous`: optimistic analytical benchmark |

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
the model's does.

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

## New dashboard: 25 history keys

Logging now selects named metrics explicitly. Adding a number to an evaluation
artifact can no longer silently create another panel. Reference values are
placed beside the corresponding model metrics, with six baseline curves total.

| Top level | Curves/fields | Count |
|---|---|---:|
| `regression` | `model_signal_mse`, `sampled_signal_mse_256`, and `query_only_signal_mse`, `ridge_quantized_signal_mse`, `bayes_continuous_signal_mse` | 5 |
| `likelihood` | `eval_answer_nll`, `train_answer_nll`, and `query_only_answer_nll`, `ridge_quantized_answer_nll`, `bayes_continuous_answer_nll` | 5 |
| `pass_exact` | `pass@1`, `pass@16`, `pass@256` on the full pool | 3 |
| `pass_sampled` | `pass@1`, `pass@16`, `pass@256` on generated prompts | 3 |
| `train` | `batch_answer_nll`, `learning_rate`, `gradient_norm_before_clip` | 3 |
| `diagnostics` | `predictive_entropy_nats`, final-only `context_shuffle_nll_increase` | 2 |
| `progress` | `optimizer_step`, `training_examples_seen`, `elapsed_seconds`, `generation_prompts` | 4 |

The shuffled-context diagnostic is shuffled-context NLL minus normal evaluation
NLL. A positive value shows that replacing the examples hurts prediction; it
does not demonstrate correct regression inference. Entropy measures predictive
spread and has no universal better direction.

There are no dashboard SEs, normal-95% bounds, per-metric counts, noisy/grid
target error families, sampling-error curves, repeated sampling settings, or
normalization/structurally fixed output-format counters. Invalid distributions
still fail validation in the evaluator. Fixed dataset/subset sizes and sampling
settings live in run config. The only evaluation-size history field is
`progress/generation_prompts`, since that size changes at the final evaluation.
Training examples seen counts presentations, including repeated pool passes.

Exact model/reference metrics use all 1,024 held-out examples. Sampled metrics
use the fixed 128-example subset periodically and all 1,024 at the final step.
Do not interpret a final jump in sampled MSE/pass as purely a model change, or
subtract full-pool exact pass from subset sampled pass as a sampling-error test.
The artifacts retain exact pass on that same subset for a valid comparison.
For a stable main regression learning curve, use `regression/model_signal_mse`.

Best NLL/step and checkpoint paths are recorded once in the run summary.
The 25-key count excludes W&B's own system metrics and these summary fields.
The optimizer, checkpoint selection (lowest evaluation NLL), numerical evaluation,
JSON/NPZ artifacts, and detailed offline reports keep their existing behavior.

This schema applies to newly started logger instances. Existing W&B histories
and workspace panels are not rewritten by a code update; the completed run
retains its old keys. New runs use the new groups; viewing an existing run more
compactly requires selecting its legacy keys manually or replaying its saved
events into a separate run with the new logger.
