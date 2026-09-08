# Evaluation metrics and normalization

This report describes the implemented evaluator and audits the completed first
run, `qwen2_1m_fixed100k_sft_10000`. The model learned a better distribution of
answers, but its predictive means did not improve on predicting zero. Both the
likelihood and signal-estimation gaps to regression references remain large.

The generation metric is now named **`sampled_mean_mse`**. Its definition remains
the MSE of the mean of 256 decoded predictions against the continuous noiseless
signal. The previous longer key has been replaced in the evaluator, W&B mapping,
tests, and the active saved evaluation records. Values and samples are unchanged.

## Evaluation protocol

Each independent problem draws `w ~ N(0,I_4/4)`, 16 context inputs and one query
from `N(0,I_4)`. Context and query noise have standard deviation 0.5. Write
`s = w·x_query` for the noiseless signal and `y = s + epsilon` for the noisy query
outcome. The visible scalar codec has 256 grid centers on `[-5,5]`, spaced
`10/255` apart. Each center is encoded as two base-16 digits.

There are 100,000 frozen training problems and 1,024 held-out evaluation problems.
Training uses one stored noisy target per problem; evaluation never draws a new
target. Step 0 and every 500 updates through step 10,000 are evaluated, giving
21 checkpoints. The best checkpoint is chosen by held-out answer NLL: step 9,500.

| Quantity | Prompts evaluated | Frequency |
|---|---:|---|
| Training-subset likelihood | Fixed 1,024 training prompts | Every evaluation |
| Held-out likelihood, exact pass@k, entropy, exact-mean errors | All 1,024 held-out prompts | Every evaluation |
| Sampled pass@k and sampled_mean_mse | Fixed 128 held-out prompts | Steps 0 through 9,500 |
| Sampled pass@k and sampled_mean_mse | All 1,024 held-out prompts | Final step 10,000 |
| Mismatched-context likelihood | All 1,024 held-out prompts | Final step only |

Generation uses 256 independent completions per selected prompt, temperature 1,
no top-k/top-p truncation, and exactly two digit tokens. The evaluator enumerates
the 16 first-digit branches and their second-digit conditional probabilities,
then samples digit pairs from those distributions on CPU. Sampling seed is
8119 + checkpoint step. Training RNG state is separate.

The completed run used a 988,032-parameter scratch Qwen2, 10,000 updates and
640,000 presentations (6.4 pool passes), with batch 64 implemented as 16×4.
It took 41.50 minutes. The current launcher uses 64×1, enables W&B, and selects
shared context/query sigma=0.1 for the follow-up experiment. The numbers in this
report describe the original sigma=0.5 run.

## 1. Should MSE be normalized?

Keep raw MSE as the primary metric. Under the implemented prior,

\[
\mathbb E[s]=0,\qquad
\operatorname{Var}(s)=\mathbb E_x[x^T(I_4/4)x]
=\mathbb E\|x\|^2/4=1.
\]

Thus MSE divided by the population signal variance is numerically identical to
raw MSE in this experiment. It would become useful for comparing experiments
with different signal scales.

A useful companion comparison is the ratio to the MSE of always predicting zero
on precisely the same prompts:

\[
R_0=\frac{\sum_i(\bar y_i-s_i)^2}{\sum_i s_i^2}.
\]

`R_0 = 0` is perfect signal prediction; 1 matches zero prediction; below 1 is an
improvement; above 1 is worse. Normalize the aggregate loss, rather than dividing
each error by `s_i^2`, which would overweight signals close to zero.

| Final evaluation quantity | Value |
|---|---:|
| Population signal variance | 1.000000 |
| Empirical mean signal | 0.057681 |
| Empirical mean squared signal / zero-prediction MSE | 1.067846 |
| Empirical centered signal variance, denominator N | 1.064519 |
| sampled_mean_mse | 1.085008 |
| Sampled-mean MSE / zero-prediction MSE | 1.016071 |
| Exact-mean MSE / zero-prediction MSE | 1.007370 |
| Sampled-mean centered R² | -0.019247 |

The baseline-relative sampled MSE is about 1.6% higher than zero prediction on
this pool. It is not a percentage error for individual predictions. Centered
R² uses `1 - sum(error²)/sum((s-mean(s))²)`, so it differs slightly from `1-R_0`.
Its constant baseline is the empirical mean signal, rather than zero.
[R² definition and interpretation](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score).

The query-noise variance, **0.25**, is not an irreducible error floor for this
metric: the target is the noiseless signal. For a predictor independent of fresh
query noise, expected continuous noisy-outcome MSE equals signal MSE plus 0.25.
That is an expectation over fresh noise, not an exact identity for this one
frozen evaluation pool, and it does not hold unchanged for quantized targets.

There is nevertheless uncertainty about the signal because the 16 context
observations are noisy. With continuous context matrix X (16×4), the posterior
weight covariance is `(4I + XᵀX/0.25)^(-1)`. The conditional Bayes signal risk is
`x_queryᵀ Cov(w|context) x_query`; it excludes the extra 0.25 for a future noisy
outcome. This follows the Bayesian linear-regression predictive distribution.
[Rasmussen and Williams, Chapter 2, equations 2.8–2.9](https://gaussianprocess.org/gpml/chapters/RW2.pdf).

On these query/context designs, mean continuous-data posterior signal variance
is **0.078878**. The continuous posterior mean has observed signal MSE **0.084507**.
These differ because posterior risk averages over latent uncertainty, whereas
observed MSE uses the one realized set of signals. This is an optimistic
continuous-data reference; the model sees quantized context and query values.

The ratios, R², and posterior-risk calculations in this section are report
diagnostics. They have not been added as separate W&B metrics. The implemented
`sampled_mean_mse` remains raw MSE with a documented target and units.

## 2. Likelihood metrics

For stored target digits `(a,b)`, answer NLL is

\[
-\log p(a\mid\mathrm{prompt})
-\log p(b\mid\mathrm{prompt},a).
\]

It is summed across the two answer tokens and then averaged across prompts,
in **nats per complete answer**. Lower is better. Prompt tokens do not contribute
to the loss, and each answer softmax contains only the 16 digit IDs.

NLL means **negative log-likelihood**. With these hard categorical targets it is
ordinary cross-entropy: the code directly calls `torch.nn.functional.cross_entropy`,
without label smoothing or class weighting. The answer loss sums two token CEs,
so it is twice the usual mean CE per answer token. Final answer NLL 4.798596
corresponds to mean answer-token CE 2.399298. This is a discrete categorical
likelihood over the quantized outcome, not a continuous Gaussian likelihood.
[PyTorch cross-entropy definition](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

The fixed training-subset answer NLL uses exactly the training CE objective on
the same 1,024 training examples at each checkpoint. The optimization log uses
the current batch of 64 examples before that optimizer update. The two losses
therefore differ in which examples and model state they measure, not in their
mathematical definition. The fixed subset helps compare training and held-out
losses across checkpoints.

| Metric | Best NLL checkpoint, 9,500 | Final checkpoint, 10,000 |
|---|---:|---:|
| Held-out answer NLL | 4.724389 ± 0.024210 | 4.798596 ± 0.014193 |
| Fixed training-subset answer NLL | 4.701619 | 4.782619 |

Uncertainty shown is one prompt SE. `answer_log_likelihood` is the negative of
`answer_nll`, so it contains the same information with the opposite direction.
Both training-subset and held-out NLL worsen at the final checkpoint. Separate
first- and second-token NLL metrics have been removed from future evaluations,
references, context controls and W&B logging. Historical records retain their
original measurements; answer-level NLL remains the likelihood metric.

## 3. Exact and generated pass@k

Let `p_i` be the model probability of the stored target's entire two-digit pair
for prompt i. Exact pass@k is the prompt average of `1-(1-p_i)^k`. It is computed
from the complete model distribution, without sampling error conditional on the
model and prompts. Average the per-prompt expression; inserting the mean p into
that expression would give a different answer for k > 1.

The stored pair encodes **the noisy outcome** `y_i = s_i + epsilon_i` after
clipping/quantization. Thus exact pass@k and generated pass@k both score the
noisy target, whereas `sampled_mean_mse` scores the continuous clean signal.
Scoring pass@k against the quantized clean signal would define a different
metric and is not what the current implementation does.

For 256 generated answers with `c_i` exact matches, the generated estimator is
`1 - C(256-c_i,k)/C(256,k)`, averaged across prompts. At k=1 it is the fraction
of generated answers matching the target; at k=256 it is the fraction of prompts
with at least one match. Higher is better. Neither quantity is greedy accuracy
or a numerical-tolerance success rate.

| k | Final exact pass@k | Final generated estimate | Generated prompt SE | Conditional sampling SD of aggregate |
|---|---:|---:|---:|---:|
| 1 | 0.008931 | 0.008900 | 0.000192 | 0.000184 |
| 2 | 0.017774 | 0.017717 | 0.000381 | 0.000364 |
| 4 | 0.035196 | 0.035106 | 0.000749 | 0.000713 |
| 8 | 0.069013 | 0.068928 | 0.001443 | 0.001371 |
| 16 | 0.132745 | 0.132910 | 0.002687 | 0.002539 |
| 32 | 0.246054 | 0.247473 | 0.004680 | 0.004381 |
| 64 | 0.426000 | 0.431518 | 0.007240 | 0.006687 |
| 128 | 0.657164 | 0.670619 | 0.009294 | 0.008560 |
| 256 | 0.861581 | 0.876953 | 0.010270 | 0.009984 |

Final generation has 2,333 matching completions out of 262,144, and at least one
match for 898 of 1,024 prompts. The generated-minus-exact pass@256 difference is
0.015372, about 1.54 conditional sampling SDs of the aggregate. It is consistent
with ordinary sampling variability.

The noisy target and fine grid make exact matching demanding. Even the
continuous Bayesian reference has exact pass@1 only 1.8922%. A high pass@256
can coexist with poor signal estimation: many broad samples can eventually hit
the stored noisy target, even if their mean barely responds to the context.

## 4. Sampled means and exact-distribution means

For each generated prompt, decode each of the 256 digit pairs to a grid center,
average those scalar predictions, and compare that mean to the unquantized
signal `s_i`:

\[
\bar y_i=\frac1{256}\sum_{j=1}^{256}\widehat y_{ij},\qquad
\mathrm{sampled\_mean\_mse}=\frac1M\sum_i(\bar y_i-s_i)^2.
\]

Final `sampled_mean_mse` is **1.085008 ± 0.060132 prompt SE**, with approximate
95% interval **[0.967149, 1.202866]**, over all 1,024 prompts. The best-NLL
checkpoint's sampled metric is **1.038493 ± 0.150797**, but uses only 128 prompts.
Do not read that difference as a change in model quality: the evaluated prompts
also change. Its same-subset zero-prediction MSE is 1.021674.

The exact-distribution mean is `mu_i = sum_b p_i(b)*center_b`, integrating all
256 possible output bins. It removes finite-generation noise and is computed on
all 1,024 prompts at every evaluation. These metrics answer related questions:
the sampled metric evaluates the actual 256-sample procedure; the exact mean
isolates the point prediction implied by the model distribution.

| Exact-mean target, final model | MSE | MAE | Signed bias, prediction minus target |
|---|---:|---:|---:|
| Continuous noiseless signal | 1.075716 | 0.773597 | -0.097151 |
| Continuous noisy outcome | 1.316209 | 0.875501 | -0.085279 |
| Decoded target grid center | 1.311072 | 0.874803 | -0.085540 |

All three target families also carry prompt SE and approximate intervals. MSE
and MAE are lower-is-better; signed bias should be interpreted relative to zero.
The noisy outcome includes the stored query noise. The decoded target additionally
includes clipping and quantization. The signal metric is the most direct of
these for assessing inference of the underlying task-specific linear function.

Conditional on model probabilities and signals, independence of the 256 draws
gives the identity

\[
\mathbb E_{\mathrm{sampling}}[(\bar y_i-s_i)^2]
=(\mu_i-s_i)^2+\operatorname{Var}_{p_i}(\widehat y)/256.
\]

For the final model, the averaged variance term is **0.008552**, giving expected
sampled MSE **1.084268**, close to the observed **1.085008**. This term is model
output variance divided by 256; it is not automatically the data noise variance
divided by 256. It is a diagnostic expectation, not another implemented metric.

For this finite answer space, routine evaluation does not require drawing
samples. Target likelihood alone gives exact pass@k; enumerating all 256 bins
also gives exact predictive means, entropy and the expected MSE of a mean of
256 independent samples through the identity above. Actual generation is useful
for checking the sampler or measuring one realized sample set. The expected
sampled-mean MSE and realized `sampled_mean_mse` are distinct quantities, so
replacing one with the other must change its metric name. The current sampler
remains enabled to produce the originally requested 256-completion diagnostic.

Paired exact-mean squared error minus zero-prediction squared error is
**+0.003939 ± 0.001478 prompt SE** at step 9,500 and
**+0.007871 ± 0.003247** at step 10,000. Final sampled error minus zero error is
**+0.017162 ± 0.007547**. Pairing is useful because both predictors face the same
signals; the raw MSE SE is not the SE of their difference. These are descriptive
comparisons on the repeatedly used evaluation pool.

## 5. Reference predictors

All rows below use the same 1,024 held-out prompts. MSE refers to each
distribution's exact grid-valued predictive mean against the continuous signal.

| Predictor | Answer NLL | Exact pass@1 | Exact pass@16 | Exact pass@256 | Signal MSE |
|---|---:|---:|---:|---:|---:|
| Best checkpoint, step 9,500 | 4.724389 | 0.010846 | 0.157105 | 0.857338 | 1.071785 |
| Final checkpoint, step 10,000 | 4.798596 | 0.008931 | 0.132745 | 0.861581 | 1.075716 |
| Uniform 256 bins | 5.545177 | 0.003906 | 0.060702 | 0.632840 | 1.067846 |
| Query-only, continuous query | 4.702490 | 0.011067 | 0.159915 | 0.860538 | 1.067846 |
| Query-only, decoded query approximation | 4.702807 | 0.011064 | 0.159876 | 0.860433 | 1.067846 |
| Bayesian, continuous data, optimistic | 4.152657 | 0.018922 | 0.256667 | 0.932689 | 0.084514 |
| Ridge/Gaussian, decoded data approximation | 4.154198 | 0.018895 | 0.256375 | 0.932440 | 0.085383 |

The query-only predictor ignores all 16 observations, using
`y|x_query ~ N(0, ||x_query||²/4 + 0.25)`. It adapts its variance to the query's
norm but always has mean zero. The continuous Bayesian predictor conditions on
the full continuous context. The decoded reference plugs quantized values into
the same Gaussian calculation; it is not the exact posterior conditioned on
quantization intervals. Gaussian mass is integrated over output bins, including
infinite endpoint tails. Uniform and both query-only distributions have zero
mean by grid symmetry and therefore the same signal MSE.

The Bayesian table value 0.084514 uses the mean of its binned distribution;
0.084507 above uses its unbinned continuous posterior mean. Their small
difference is expected from binning and clipping.

For a direct comparison with simple regression, the following methods were also
evaluated on the same 1,024 frozen problems. Each method fits only that problem's
16 context observations, then predicts its query; there is no fitted intercept.
These are unbinned scalar point predictions. The decoded rows use exactly the
grid-valued observations and query accessible through the model's tokens.

| Point predictor | Inputs/observations | Clean-signal MSE | Clean MSE prompt SE | Continuous noisy-outcome MSE |
|---|---|---:|---:|---:|
| Bayesian posterior mean / ridge, lambda=1 | Continuous | 0.084507 | 0.005052 | 0.370482 |
| Ridge, lambda=1 | Decoded | 0.085375 | 0.005089 | 0.371591 |
| Ordinary least squares | Continuous | 0.093552 | 0.005296 | 0.382866 |
| Ordinary least squares | Decoded | 0.094482 | 0.005355 | 0.384057 |

The known prior `w ~ N(0,I/4)` and noise variance 0.25 imply posterior mean
`w_hat = (XᵀX + I)^(-1)Xᵀy`. This is ridge regression with lambda=1 under the
objective `||y-Xw||² + lambda*||w||²`; lambda would be 1/16 if the data-fit term
were instead divided by the 16 observations. Ordinary least squares omits that
penalty. The implementation used the existing Cholesky Bayesian solver and
NumPy's least-squares solver, respectively; all context matrices had rank 4.
[Ridge objective convention](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

These additional point-predictor results are report diagnostics, not new
periodically logged reference distributions. The decoded ridge and ordinary
least-squares results show that simple per-prompt fitting achieves clean-signal
MSE around 0.085–0.094 with the token-visible data, versus about 1.07 for the
model's exact predictive mean. Fresh query noise would add 0.25 to expected
continuous-outcome MSE; the finite frozen-pool differences need not equal 0.25.

The selected model's paired NLL gap is **+0.021899 ± 0.009855 prompt SE** versus
continuous query-only and **+0.571732 ± 0.028673** versus continuous Bayesian.
Thus the likelihood improvement from untrained NLL 5.722497 has not translated
into useful signal prediction relative to zero. The roughly 12.7-fold exact-mean
MSE gap between the best model and the Bayesian reference is much larger than
the continuous-versus-decoded reference gap.

## 6. Entropy, context control, validity, and data audits

Answer entropy is `-sum_b p_i(b) log p_i(b)`, averaged across prompts. It measures
distribution spread, not accuracy or calibration. It is 4.726965 nats at the best
checkpoint and 5.012964 at the final checkpoint, compared with 5.545177 for uniform
and 4.092616 for the continuous Bayesian reference. The final distribution is
broader while NLL worsens. Entropy alone is not a target to minimize.

The final context control cyclically shifts complete observation blocks between
prompts, preserving each query and stored target. NLL rises from **4.798596** to
**4.818172**, an observed increase of **0.019576**. This shows sensitivity to
context, but does not establish useful or optimal inference. No paired SE of
this control difference is currently saved.

Generation has **zero invalid completions** and **zero overlength completions**.
These primarily check implementation integrity because sampling is restricted
to digit tokens and explicitly produces exactly two tokens. They are not tests
of unrestricted language-generation formatting ability. Maximum joint
probability normalization error is **2.28e-7**; errors above 1e-6 fail evaluation.

| Split | Context-input clipping | Context-outcome clipping | Query-input clipping | Query-outcome clipping |
|---|---:|---:|---:|---:|
| Train | 6 / 6,400,000 | 627 / 1,600,000 | 1 / 400,000 | 39 / 100,000 |
| Evaluation | 0 / 65,536 | 9 / 16,384 | 0 / 4,096 | 1 / 1,024 |

Counts refer to continuous values outside `[-5,5]`. Prompt hashes find zero
overlap between train and evaluation; dataset file and array hashes verify the
frozen pool. These checks do not create an independent test split.

## 7. Uncertainty, logging, and limits

The uncertainty fields below remain in JSON/NPZ artifacts and detailed offline
reports. The current W&B dashboard uses an explicit 25-key selection without
SEs, intervals or per-metric counts. See [METRICS.md](METRICS.md) for the current
layout, target/reference explanations, and the audit of the old 827-key history.

For any per-prompt metric vector, `prompt_se` is its sample standard deviation
(ddof=1) divided by the square root of prompt count. `prompt_normal95` is
mean ± 1.96 SE. These are approximate intervals; checkpoint comparisons share
the same examples and the best checkpoint was chosen on this pool.

For generated pass@k, the evaluator additionally integrates the combinatorial
estimator over `C_i ~ Binomial(256,p_i)` to obtain the conditional sampling SD
of the aggregate. This holds model and prompts fixed. Prompt SE and conditional
sampling SD describe different sources of variability and should not simply
be added together. `sampled_minus_exact` stores paired per-prompt differences;
`exact_pass_same_subset` ensures the generated comparison uses the same prompts.

The 256 samples within a prompt are not 256 independent regression problems.
The sampled-MSE SE uses one squared error per prompt. It includes the variation
in the realized sampled estimator across prompts but is not a separately
estimated conditional Monte Carlo SD. None of these uncertainties includes
training-seed variability or corrects for checkpoint selection.

The following mapping describes the **historical** recursive W&B logger. It
helps read existing runs; new logger instances use the schema in METRICS.md.

| Local event field | Historical W&B scalar key |
|---|---|
| `eval.answer_nll.mean` | `eval/answer_nll` |
| `eval.exact_pass[k].mean` | `eval/exact_pass@k` |
| `eval.generation.generative_pass[k].mean` | `eval/generation/generative_pass@k` |
| `eval.generation.sampled_mean_mse.mean` | `eval/generation/sampled_mean_mse` |
| `eval.predictive_mean_errors[target].mse.mean` | `eval/predictive_mean_errors/target/mse` |
| `eval.entropy_nats_per_answer.mean` | `eval/entropy_nats_per_answer` |
| `train.answer_nll.mean` in evaluation events | `train_eval/answer_nll` |

Replace `k` and `target` in the mapping with the actual value or target-family
name. SE, interval bounds and counts use suffixes beneath the same key. Other
scalar diagnostics are flattened similarly, and references use `reference/`.
Optimization events separately record `train/answer_nll`, learning rate,
gradient norm before clipping, global step, presentations and elapsed time.
The optimization loss is from the current batch during the update; the fixed
training-subset NLL is a separate evaluation measurement.

The original run used local logging; its metric additions and rename do not
upload it to W&B. Future training through the current launcher logs online.
Source metrics are in `outputs/first_run/metrics.jsonl` and the copied best/final
checkpoint JSON files. Full probability tables, sampled digit pairs and all
21 checkpoints remain in the corresponding server run directory.

For judging regression learning, prioritize exact-mean signal MSE and the
256-sample mean MSE alongside query-only and Bayesian references. Use answer
NLL to assess probabilistic prediction, and exact pass@k to characterize sampled
target matching. This run provides no independent final test, multiple training
seeds, or evidence that the architecture has reached its representational limit.
The current suite also does not measure posterior calibration/coverage or
performance across new data distributions.
