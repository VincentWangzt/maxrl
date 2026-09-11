"""Focused population-loss and end-to-end gradient checks; execute on the server."""

import json
import os
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from noisy_regression.codec import CENTERS, DIGITS
from noisy_regression.data import DatasetConfig, FrozenOrder, generate_split, prepare
from noisy_regression.evaluate import evaluate, select_device
from noisy_regression.model import (
    ModelConfig,
    conditional_log_probs,
    create_model,
    joint_log_probs,
    make_optimizer,
    make_scheduler,
)
from noisy_regression.population import PopulationConfig, population_loss
from noisy_regression.tracking import event_metrics
from noisy_regression.train import TrainConfig, optimize_step, train


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)
    torch.manual_seed(24)


def tiny_model():
    return create_model(
        ModelConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )


@pytest.mark.parametrize(
    "method,degree",
    [("rloo", None), ("grpo", None), ("maxrl", 1), ("maxrl", 3), ("maxrl", 256), ("maxrl", "inf")],
)
def test_loss_matches_independent_population_gradient(method, degree):
    first = torch.randn(3, DIGITS, dtype=torch.double, requires_grad=True)
    second = torch.randn(3, DIGITS, DIGITS, dtype=torch.double, requires_grad=True)
    log_probs = joint_log_probs(first.log_softmax(-1), second.log_softmax(-1))
    targets = torch.tensor([[0, 0], [8, 2], [15, 15]])
    config = PopulationConfig(method, degree, maxrl_tau=0.2)
    loss, metrics = population_loss(log_probs, targets, config)
    probabilities = log_probs.exp()
    centers = torch.tensor(CENTERS)
    errors = (centers - centers[targets[:, 0] * DIGITS + targets[:, 1], None]).square()
    expected_mse = (probabilities * errors).sum(-1)
    if method == "rloo":
        direct = expected_mse
    elif method == "grpo":
        std = (probabilities * (errors - expected_mse[:, None]).square()).sum(-1).sqrt()
        direct = expected_mse / (std.detach() + config.grpo_epsilon)
    else:
        z = (probabilities * (-errors / (2 * config.maxrl_tau**2)).exp()).sum(-1)
        direct = -z.log() if degree == "inf" else sum((1 - z).pow(k) / k for k in range(1, degree + 1))
    actual_grad = torch.autograd.grad(loss.mean(), (first, second), retain_graph=True)
    expected_grad = torch.autograd.grad(direct.mean(), (first, second))
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-8)
    if method != "grpo":
        torch.testing.assert_close(loss, direct)
    torch.testing.assert_close(
        metrics["expected_mse"], metrics["predictive_mean_mse"] + metrics["predictive_variance"]
    )


@pytest.mark.parametrize("degree", [256, "inf"])
def test_maxrl_tiny_likelihood_does_not_clip_gradient(degree):
    logits = torch.full((1, 256), -2000.0, dtype=torch.double)
    logits[0, -1] = 0
    logits.requires_grad_()
    loss, metrics = population_loss(
        logits.log_softmax(-1), torch.tensor([[0, 0]]), PopulationConfig("maxrl", degree, maxrl_tau=0.001)
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all()
    if degree == "inf":
        assert loss.item() == pytest.approx(2000)
        assert logits.grad[0, 0].item() == pytest.approx(-1)
        assert logits.grad[0, -1].item() == pytest.approx(1)
    else:
        assert loss.item() == pytest.approx(sum(1 / k for k in range(1, 257)))
    assert metrics["maxrl_log_expected_reward"].item() == pytest.approx(-2000)


def test_cached_joint_values_and_parameter_gradients_match_all_256_full_sequences():
    arrays = generate_split(DatasetConfig(train_count=1, eval_count=1, observations=1), "train")
    prompts = torch.tensor(arrays["tokens"][:, :-3].astype(np.int64))
    model = tiny_model()
    cached = joint_log_probs(*conditional_log_probs(model, prompts))
    a = torch.arange(256) // 16
    b = torch.arange(256) % 16
    full = torch.cat([prompts.repeat_interleave(256, 0), a[:, None], b[:, None]], dim=1)
    logits = model(input_ids=full, use_cache=False).logits.float()
    rows = torch.arange(256)
    brute = (
        (logits[:, -3, :16].log_softmax(-1)[rows, a] + logits[:, -2, :16].log_softmax(-1)[rows, b])
        .double()
        .reshape(1, 256)
    )
    torch.testing.assert_close(cached, brute, atol=1e-6, rtol=1e-6)
    weights = torch.randn_like(cached)
    parameters = tuple(model.parameters())
    actual = torch.autograd.grad((cached * weights).sum(), parameters)
    expected = torch.autograd.grad((brute * weights).sum(), parameters)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=5e-5, rtol=2e-4)


@pytest.mark.parametrize("method,degree", [("grpo", None), ("rloo", None), ("maxrl", 256), ("maxrl", "inf")])
def test_training_accumulation_evaluation_and_logging(method, degree):
    arrays = generate_split(DatasetConfig(train_count=4, eval_count=2, observations=1), "train")
    model, other = tiny_model(), tiny_model()
    other.load_state_dict(model.state_dict())
    config = TrainConfig(
        method=method,
        maxrl_degree=degree,
        batch_size=4,
        micro_batch_size=4,
        max_steps=2,
        warmup_steps=0,
        device="cpu",
        precision="fp32",
    )
    order = FrozenOrder(4)
    state = order.state_dict()
    results = []
    for current, microbatch in ((model, 4), (other, 2)):
        optimizer, _ = make_optimizer(current, 1e-4, 0.9, 0.95, 0.01, 1e-8)
        scheduler = make_scheduler(optimizer, 0, 2, 0, "linear_warmup_constant", warmup_start_factor=0.1)
        local_order = FrozenOrder(4)
        local_order.load_state_dict(state)
        metrics, _ = optimize_step(
            current,
            optimizer,
            scheduler,
            local_order,
            arrays["tokens"],
            replace(config, micro_batch_size=microbatch),
            torch.device("cpu"),
        )
        assert metrics["gradient_norm"] > 0 and "eos_nll" not in metrics
        results.append(metrics)
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    assert results[0]["population"]["loss"] == pytest.approx(results[1]["population"]["loss"], abs=1e-6)
    report = evaluate(
        model, arrays, 2, torch.device("cpu"), "fp32", population_config=config.population_config()
    )
    event = {"kind": "evaluation", "eval": report, "step": 1, "elapsed_seconds": 0, "evaluation_seconds": 0}
    logged = event_metrics(event)
    assert "eval/population/predictive_variance" in logged
    assert "eval/mse_over_target_variance/noisy" in logged


def test_population_train_checkpoint_resume(tmp_path):
    data = tmp_path / "data"
    prepare(data, DatasetConfig(train_count=4, eval_count=2, observations=1))
    model_config = ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    config = TrainConfig(
        method="maxrl",
        maxrl_degree="inf",
        batch_size=2,
        micro_batch_size=1,
        max_steps=1,
        warmup_steps=0,
        device="cpu",
        precision="fp32",
        eval_batch_size=1,
    )
    first = train(data, tmp_path / "first", config, model_config)
    resumed = train(
        data,
        tmp_path / "resumed",
        replace(config, max_steps=2),
        model_config,
        resume=first["final_checkpoint"],
    )
    assert resumed["steps"] == 2 and resumed["presentations"] == 4


@pytest.mark.parametrize(
    "method,degree", [("maxrl", None), ("maxrl", 0), ("maxrl", -1), ("maxrl", 2.5), ("grpo", 256)]
)
def test_invalid_degree_rejected(method, degree):
    with pytest.raises(ValueError):
        PopulationConfig(method, degree)


@pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="Requires an explicitly selected free GPU"
)
def test_cuda_full_batch_smoke():
    device = select_device("cuda:0", "bf16")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    arrays = generate_split(DatasetConfig(train_count=1024, eval_count=1), "train")
    model = create_model(ModelConfig()).to(device)
    optimizer, _ = make_optimizer(model, 1e-4, 0.9, 0.95, 0.01, 1e-8)
    scheduler = make_scheduler(optimizer, 200, 20_000, 0, "linear_warmup_constant", warmup_start_factor=0.1)
    config = TrainConfig(method="maxrl", maxrl_degree="inf", batch_size=1024, micro_batch_size=256)
    started = time.perf_counter()
    metrics, _ = optimize_step(
        model, optimizer, scheduler, FrozenOrder(1024), arrays["tokens"], config, device
    )
    assert metrics["gradient_norm"] > 0
    assert all(np.isfinite(value) for value in metrics["population"].values())
    print(
        json.dumps(
            {
                "seconds": time.perf_counter() - started,
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                **metrics,
            }
        )
    )
