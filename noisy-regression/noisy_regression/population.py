"""Exact population objectives over the 256 autoregressive two-digit answers.

Targets are decoded observed answer tokens, as in nanochat's scalar objectives.
RLOO/GRPO reward is negative squared error; MaxRL reward is exp(-error²/(2τ²)).
These are on-policy population gradients, with no rollout sampling or PPO reuse.
"""

import math
from dataclasses import dataclass

import torch

from noisy_regression.codec import CENTERS, DIGITS


def parse_degree(value):
    if value == "inf":
        return "inf"
    degree = int(value)
    if degree < 1:
        raise ValueError("MaxRL degree must be a positive integer or 'inf'")
    return degree


@dataclass(frozen=True)
class PopulationConfig:
    method: str
    maxrl_degree: int | str | None = None
    maxrl_tau: float = 0.1
    grpo_epsilon: float = 1e-8

    def __post_init__(self):
        if self.method not in ("grpo", "rloo", "maxrl"):
            raise ValueError("Population method must be grpo, rloo or maxrl")
        if self.method == "maxrl":
            if self.maxrl_degree != "inf" and (type(self.maxrl_degree) is not int or self.maxrl_degree < 1):
                raise ValueError("MaxRL requires an explicit positive integer degree or 'inf'")
        elif self.maxrl_degree is not None:
            raise ValueError("maxrl_degree is only valid for MaxRL")
        if not math.isfinite(self.maxrl_tau) or self.maxrl_tau <= 0:
            raise ValueError("MaxRL tau must be finite and positive")
        if not math.isfinite(self.grpo_epsilon) or self.grpo_epsilon <= 0:
            raise ValueError("GRPO epsilon must be finite and positive")


def population_loss(log_probs, target_digits, config):
    """Return per-prompt loss and detached diagnostics (all exact expectations).

    MaxRL degree d minimizes sum_{k=1}^d (1-Z)^k/k; degree inf minimizes
    -log Z, where Z=E_p[exp(-error²/(2τ²))]. Its gradient uses the normalized
    reward posterior in log space, avoiding an artificial floor on Z. Finite
    degree multiplies that gradient by 1-(1-Z)^d. GRPO detaches both population
    moments and score weights: differentiating through them is a different loss.
    """
    if log_probs.ndim != 2 or log_probs.shape[1] != DIGITS**2:
        raise ValueError("Expected every two-digit combination, shape (batch, 256)")
    if target_digits.shape != (len(log_probs), 2) or target_digits.dtype != torch.long:
        raise ValueError("Expected integer target digits with shape (batch, 2)")
    if torch.any((target_digits < 0) | (target_digits >= DIGITS)):
        raise ValueError("Target digits must be in [0, 15]")
    # The table is small; FP64 keeps moments and small likelihoods stable.
    log_probs = log_probs.double()
    if not torch.isfinite(log_probs).all() or not torch.allclose(
        log_probs.logsumexp(-1),
        torch.zeros(len(log_probs), device=log_probs.device, dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("Expected a finite normalized joint log distribution")
    centers = torch.as_tensor(CENTERS, device=log_probs.device)
    indices = target_digits[:, 0] * DIGITS + target_digits[:, 1]
    targets = centers[indices]
    probabilities = log_probs.exp()
    squared_errors = (centers[None, :] - targets[:, None]).square()
    expected_mse = (probabilities * squared_errors).sum(-1)
    with torch.no_grad():
        mean = (probabilities * centers).sum(-1)
        variance = (probabilities * (centers - mean[:, None]).square()).sum(-1)
        reward_std = (probabilities * (squared_errors - expected_mse[:, None]).square()).sum(-1).sqrt()
        metrics = {
            "expected_mse": expected_mse.detach(),
            "predictive_mean_mse": (mean - targets).square(),
            "predictive_variance": variance,
            "reward_mean": -expected_mse.detach(),
            "reward_std": reward_std,
            "entropy_nats": -(probabilities * log_probs).sum(-1),
            "answer_nll": -log_probs.gather(1, indices[:, None]).squeeze(1),
        }
    if config.method == "rloo":
        loss = expected_mse
    elif config.method == "grpo":
        advantages = (expected_mse[:, None] - squared_errors) / (reward_std[:, None] + config.grpo_epsilon)
        loss = -((probabilities * advantages).detach() * log_probs).sum(-1)
    else:
        with torch.no_grad():
            log_rewards = -squared_errors / (2 * config.maxrl_tau**2)
            log_z = (log_probs + log_rewards).logsumexp(-1)
            z = log_z.exp().clamp(max=1)
            posterior = (log_probs + log_rewards - log_z[:, None]).exp()
            if config.maxrl_degree == "inf":
                scale = torch.ones_like(z)
                objective = -log_z
            else:
                scale = -torch.expm1(config.maxrl_degree * torch.log1p(-z))
                powers = torch.arange(1, config.maxrl_degree + 1, device=z.device, dtype=z.dtype)
                objective = ((1 - z[:, None]).pow(powers) / powers).sum(-1)
            weights = scale[:, None] * (posterior - probabilities)
            metrics.update(
                {
                    "maxrl_expected_reward": z,
                    "maxrl_log_expected_reward": log_z,
                    "maxrl_gradient_scale": scale,
                }
            )
        surrogate = -(weights * log_probs).sum(-1)
        loss = surrogate - surrogate.detach() + objective
    metrics["loss"] = loss.detach()
    return loss, metrics
