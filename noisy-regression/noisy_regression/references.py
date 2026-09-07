"""Continuous-data optimistic and decoded-data approximate Gaussian references."""

import numpy as np
from scipy.special import log_ndtr

from noisy_regression.codec import MIDPOINTS, decode, encode
from noisy_regression.metrics import distribution_summary


def gaussian_bin_log_probs(mean, variance):
    mean, variance = np.asarray(mean), np.asarray(variance)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("Gaussian means must be finite and variances positive")
    boundaries = np.concatenate(([-np.inf], MIDPOINTS, [np.inf]))
    z = (boundaries[None, :] - mean[:, None]) / np.sqrt(variance[:, None])
    lower, upper = z[:, :-1], z[:, 1:]
    # Use survival probabilities in the positive tail to avoid subtracting
    # CDFs rounded to one. log(-expm1(...)) retains narrow-bin precision.
    log_large = np.where(lower >= 0, log_ndtr(-lower), log_ndtr(upper))
    log_small = np.where(lower >= 0, log_ndtr(-upper), log_ndtr(lower))
    return log_large + np.log(-np.expm1(log_small - log_large))


def bayesian_predictive(context_x, context_y, query_x, sigma=0.5):
    dimension = context_x.shape[-1]
    precision = dimension * np.eye(dimension) + np.swapaxes(context_x, -1, -2) @ context_x / sigma**2
    chol = np.linalg.cholesky(precision)
    rhs = np.einsum("bnd,bn->bd", context_x, context_y) / sigma**2
    intermediate = np.linalg.solve(chol, rhs[..., None])
    posterior_mean = np.linalg.solve(chol.swapaxes(-1, -2), intermediate)[..., 0]
    projected = np.linalg.solve(chol, query_x[..., None])[..., 0]
    return np.einsum("bd,bd->b", query_x, posterior_mean), sigma**2 + np.square(projected).sum(-1)


def reference_distributions(arrays):
    count = len(arrays["tokens"])
    result = {"uniform_256": np.full((count, 256), -np.log(256))}
    for decoded in (False, True):
        x, y, query = arrays["context_x"], arrays["context_y"], arrays["query_x"]
        if decoded:
            x, y, query = decode(encode(x)), decode(encode(y)), decode(encode(query))
        query_name = "query_only_decoded_plugin_approximation" if decoded else "query_only_continuous_optimistic"
        result[query_name] = gaussian_bin_log_probs(np.zeros(count), 0.5**2 + (query**2).sum(-1) / 4)
        mean, variance = bayesian_predictive(x, y, query)
        bayes_name = "ridge_decoded_gaussian_approximation" if decoded else "bayesian_continuous_optimistic"
        result[bayes_name] = gaussian_bin_log_probs(mean, variance)
    return result


def reference_report(arrays):
    return {
        name: distribution_summary(log_probs, arrays) for name, log_probs in reference_distributions(arrays).items()
    }
