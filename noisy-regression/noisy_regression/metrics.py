"""Answer-level likelihood, pass@k, diagnostics, and task-level uncertainty."""

import numpy as np
from scipy.stats import binom

from noisy_regression.codec import CENTERS, PROMPT_LENGTH, decode, quantize

KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite vector")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "prompt_se": se,
        "prompt_normal95": [mean - 1.96 * se, mean + 1.96 * se],
        "prompts": len(values),
    }


def exact_pass(probabilities, k):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if k < 1 or np.any((probabilities < 0) | (probabilities > 1)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probabilities or k")
    with np.errstate(divide="ignore"):
        return -np.expm1(k * np.log1p(-probabilities))


def estimated_pass(counts, n, k):
    counts = np.asarray(counts)
    if counts.dtype.kind not in "iu" or np.any((counts < 0) | (counts > n)) or not 1 <= k <= n:
        raise ValueError("Expected integer 0 <= c <= N and 1 <= k <= N")
    # Product over k; zero factors cover c=0 and c>N-k without log singularities.
    failures = np.ones(counts.shape, dtype=np.float64)
    for offset in range(k):
        failures *= np.maximum(n - counts - offset, 0) / (n - offset)
    return 1.0 - failures


def sampled_summary(counts, target_probs, n):
    result = {
        "completion_count_per_prompt": n,
        "prompts": len(counts),
        "success_count": int(np.sum(counts)),
        "generative_pass": {},
        "exact_pass_same_subset": {},
        "sampled_minus_exact": {},
    }
    # Given the model probabilities, C_i ~ Binomial(N,p_i). Integrating the
    # estimator over C quantifies sampling variability without pretending that
    # completions are additional independent regression problems.
    possible_counts = np.arange(n + 1)
    mass = binom.pmf(possible_counts[None, :], n, np.asarray(target_probs)[:, None])
    for k in KS:
        exact = exact_pass(target_probs, k)
        sampled = estimated_pass(counts, n, k)
        all_estimates = estimated_pass(possible_counts, n, k)
        expected = mass @ all_estimates
        variances = np.maximum(mass @ all_estimates**2 - expected**2, 0)
        entry = mean_se(sampled)
        entry["conditional_sampling_sd_of_mean"] = float(np.sqrt(variances.sum()) / len(counts))
        result["generative_pass"][str(k)] = entry
        result["exact_pass_same_subset"][str(k)] = mean_se(exact)
        result["sampled_minus_exact"][str(k)] = mean_se(sampled - exact)
    return result


def sampled_mean_mse(completions, noiseless_signal):
    """Decode 256 samples per prompt, average them, then score against w·x_query."""
    completions = np.asarray(completions)
    noiseless_signal = np.asarray(noiseless_signal, dtype=np.float64)
    if (
        noiseless_signal.ndim != 1
        or len(noiseless_signal) < 1
        or not np.isfinite(noiseless_signal).all()
        or completions.shape != (len(noiseless_signal), 256, 2)
    ):
        raise ValueError("Expected (prompts, 256, 2) completions and one finite noiseless signal per prompt")
    sampled_mean = decode(completions).mean(axis=1)
    return mean_se((sampled_mean - noiseless_signal) ** 2)


def distribution_summary(log_probs, arrays):
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.shape != (len(arrays["tokens"]), 256) or not np.isfinite(log_probs).all():
        raise ValueError("Expected finite (prompts, 256) log probabilities")
    probs = np.exp(log_probs)
    normalization_error = float(np.abs(probs.sum(1) - 1).max())
    if normalization_error > 1e-6:
        raise ValueError(f"Distribution normalization error {normalization_error}")
    targets = arrays["tokens"][:, PROMPT_LENGTH : PROMPT_LENGTH + 2].astype(np.int64)
    target_indices = 16 * targets[:, 0] + targets[:, 1]
    rows = np.arange(len(targets))
    target_ll = log_probs[rows, target_indices]
    # Clean likelihood/pass score the quantized signal under the SAME model
    # distribution. MSE below keeps both clean and noisy targets continuous.
    clean_target_ll = log_probs[rows, quantize(arrays["query_signal"])]
    predictive_mean = probs @ CENTERS
    result = {
        "prompts": len(targets),
        "answer_nll": mean_se(-target_ll),
        "clean_answer_nll": mean_se(-clean_target_ll),
        "answer_log_likelihood": mean_se(target_ll),
        "entropy_nats_per_answer": mean_se(-(probs * log_probs).sum(1)),
        "max_normalization_error": normalization_error,
        "exact_pass": {str(k): mean_se(exact_pass(np.exp(target_ll), k)) for k in KS},
        "clean_exact_pass": {str(k): mean_se(exact_pass(np.exp(clean_target_ll), k)) for k in KS},
        "predictive_mean_errors": {},
    }
    for name, target in (
        ("continuous_noisy_outcome", arrays["query_y"]),
        ("decoded_target_grid_value", decode(targets)),
        ("continuous_noiseless_signal", arrays["query_signal"]),
    ):
        errors = predictive_mean - target
        result["predictive_mean_errors"][name] = {
            "mse": mean_se(errors**2),
            "mae": mean_se(np.abs(errors)),
            "bias": mean_se(errors),
        }
    return result
