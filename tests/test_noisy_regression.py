"""Focused CPU checks. Run only on cmu-L40-live via noisy-regression/validate.sh."""

import json
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from noisy_regression.codec import (
    ANSWER,
    BOS,
    CENTERS,
    DELTA,
    MIDPOINTS,
    PROMPT_LENGTH,
    QUERY,
    SEQUENCE_LENGTH,
    X,
    Y,
    build_sequences,
    decode,
    encode,
    quantize,
)
from noisy_regression.data import DatasetConfig, FrozenOrder, array_hash, generate_split, load_pool, prepare
from noisy_regression.evaluate import evaluate
from noisy_regression.evaluate_baseline import evaluate_baseline
from noisy_regression.metrics import (
    distribution_summary,
    estimated_pass,
    exact_pass,
    sampled_mean_mse,
    sampled_summary,
)
from noisy_regression.model import (
    ModelConfig,
    answer_labels,
    answer_nll_from_logits,
    conditional_log_probs,
    create_model,
    joint_log_probs,
    make_optimizer,
    make_scheduler,
    sample_answers,
    teacher_forced_nll,
)
from noisy_regression.references import bayesian_predictive, gaussian_bin_log_probs, reference_distributions
from noisy_regression.tracking import TrackingConfig, event_metrics
from noisy_regression.train import TrainConfig, load_checkpoint, optimize_step, save_checkpoint, train
from scipy.stats import binom


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)


@pytest.fixture
def arrays():
    return generate_split(DatasetConfig(train_count=8, eval_count=4), "train")


def test_codec_endpoints_midpoints_roundtrips_and_finite():
    np.testing.assert_array_equal(quantize(CENTERS), np.arange(256))
    np.testing.assert_array_equal(decode(encode(CENTERS)), CENTERS)
    np.testing.assert_array_equal(quantize(MIDPOINTS), np.arange(1, 256))
    np.testing.assert_array_equal(quantize(np.nextafter(MIDPOINTS, -np.inf)), np.arange(255))
    np.testing.assert_array_equal(quantize(np.nextafter(MIDPOINTS, np.inf)), np.arange(1, 256))
    np.testing.assert_array_equal(quantize([-1e300, -5, 0, 5, 1e300]), [0, 0, 128, 255, 255])
    assert DELTA == 10 / 255 and not np.any(CENTERS == 0)
    np.testing.assert_array_equal(encode([-5, 5]), [[0, 0], [15, 15]])
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="nonfinite"):
            encode(value)
    for pair in ([0, 16], [-1, 0], [0.0, 1.0], [1], 1):
        with pytest.raises(ValueError):
            decode(pair)
    rng = np.random.default_rng(100)
    z = rng.uniform(-10, 10, 10000)
    np.testing.assert_array_equal(quantize(z), np.clip(np.floor((z + 5) / DELTA + 0.5), 0, 255))


def test_frozen_reproducible_splits_and_hashes(tmp_path):
    config = DatasetConfig(train_count=8, eval_count=4)
    train_arrays, eval_arrays = generate_split(config, "train"), generate_split(config, "eval")
    assert array_hash(train_arrays) == array_hash(generate_split(config, "train"))
    assert array_hash(eval_arrays) == array_hash(generate_split(config, "eval"))
    assert not set(train_arrays["prompt_hashes"]) & set(eval_arrays["prompt_hashes"])
    np.testing.assert_array_equal(
        train_arrays["context_y"],
        np.einsum("bnd,bd->bn", train_arrays["context_x"], train_arrays["w"]) + train_arrays["context_noise"],
    )
    np.testing.assert_array_equal(train_arrays["query_y"], train_arrays["query_signal"] + train_arrays["query_noise"])
    changed = generate_split(replace(config, train_count=20), "eval")
    assert array_hash(changed) == array_hash(eval_arrays)
    directory = tmp_path / "pool"
    metadata = prepare(directory, config)
    pools, loaded_metadata = load_pool(directory)
    assert loaded_metadata == metadata
    assert array_hash(pools["train"]) == array_hash(train_arrays)
    assert not pools["train"]["query_y"].flags.writeable
    assert set(metadata["splits"]) == {"train", "eval"}
    with pytest.raises(FileExistsError):
        prepare(directory, config)
    with (directory / "train.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="modified"):
        load_pool(directory)
    with pytest.raises(ValueError):
        generate_split(replace(config, eval_seed=config.train_seed), "train")


def test_prompt_layout_and_no_latent_leakage(arrays):
    tokens = arrays["tokens"]
    assert tokens.shape == (8, 205)
    assert (tokens[:, 0] == BOS).all() and (tokens[:, 193] == QUERY).all() and (tokens[:, 202] == ANSWER).all()
    for i in range(16):
        offset = 1 + 12 * i
        assert (tokens[:, offset] == X).all() and (tokens[:, offset + 9] == Y).all()
        np.testing.assert_array_equal(
            tokens[:, offset + 1 : offset + 9], encode(arrays["context_x"][:, i]).reshape(8, 8)
        )
        np.testing.assert_array_equal(tokens[:, offset + 10 : offset + 12], encode(arrays["context_y"][:, i]))
    np.testing.assert_array_equal(tokens[:, 194:202], encode(arrays["query_x"]).reshape(8, 8))
    np.testing.assert_array_equal(tokens[:, -2:], encode(arrays["query_y"]))
    replaced_target = build_sequences(
        arrays["context_x"], arrays["context_y"], arrays["query_x"], arrays["query_y"] + 1
    )
    np.testing.assert_array_equal(tokens[:, :203], replaced_target[:, :203])
    with pytest.raises(ValueError, match="truncation"):
        build_sequences(arrays["context_x"], arrays["context_y"], arrays["query_x"], arrays["query_y"], 204)


def test_shared_noise_setting_preserves_latents_and_matches_bayesian_covariance():
    config = DatasetConfig(train_count=8, eval_count=4)
    baseline = generate_split(config, "eval")
    for sigma in (0.1, 0.2):
        changed = generate_split(replace(config, sigma=sigma), "eval")
        for name in ("w", "context_x", "query_x", "query_signal", "ids"):
            np.testing.assert_array_equal(changed[name], baseline[name])
        for name in ("context_noise", "query_noise"):
            np.testing.assert_allclose(changed[name], baseline[name] * (sigma / config.sigma))
        assert not np.array_equal(changed["tokens"][:, :193], baseline["tokens"][:, :193])
        np.testing.assert_array_equal(changed["tokens"][:, 193:203], baseline["tokens"][:, 193:203])
        assert not np.array_equal(changed["tokens"][:, -2:], baseline["tokens"][:, -2:])

    context_x = np.tile(np.eye(4), (1, 4, 1))
    weights = np.array([0.2, -0.3, 0.4, 0.1])
    context_y = context_x @ weights
    query_x = np.ones((1, 4))
    for sigma in (0.5, 0.1):
        mean, variance = bayesian_predictive(context_x, context_y, query_x, sigma)
        np.testing.assert_allclose(mean, [weights.sum() / (1 + sigma**2)])
        np.testing.assert_allclose(variance, [sigma**2 + sigma**2 / (1 + sigma**2)])
    low_noise_config = replace(config, sigma=0.1)
    lower = reference_distributions(baseline, low_noise_config)
    original = reference_distributions(baseline, config)
    assert not np.allclose(lower["query_only_continuous_optimistic"], original["query_only_continuous_optimistic"])
    assert not np.allclose(lower["bayesian_continuous_optimistic"], original["bayesian_continuous_optimistic"])
    for value in (-0.1, np.nan, 0.0, np.inf):
        with pytest.raises(ValueError, match="sigma"):
            replace(config, sigma=value).validate()


def test_epoch_order_freezes_complete_examples(arrays):
    stream = FrozenOrder(8, 19)
    first, second = stream.take(8), stream.take(8)
    assert set(first) == set(second) == set(range(8))
    assert not np.array_equal(first, second)
    before = array_hash(arrays)
    stream.take(23)
    assert stream.presentations == 39
    state = stream.state_dict()
    expected = stream.take(29)
    restored = FrozenOrder(8, 999)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(expected, restored.take(29))
    assert array_hash(arrays) == before


def test_answer_only_shift_and_restricted_loss(arrays):
    tokens = torch.tensor(arrays["tokens"].astype(np.int64))
    labels = answer_labels(tokens)
    assert (labels[:, :-2] == -100).all() and torch.equal(labels[:, -2:], tokens[:, -2:])
    logits = torch.zeros(8, SEQUENCE_LENGTH, 22, requires_grad=True)
    nll = answer_nll_from_logits(logits, tokens)
    torch.testing.assert_close(nll.sum(1), torch.full((8,), math.log(256)))
    nll.sum(1).mean().backward()
    assert torch.count_nonzero(logits.grad[:, :202]) == 0
    assert torch.count_nonzero(logits.grad[:, 204:]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 16:]) == 0
    altered = logits.detach().clone()
    altered[:, :, 16:] = 1e6
    altered[:, :202] = -1e6
    altered[:, 204:] = 1e6
    torch.testing.assert_close(answer_nll_from_logits(altered, tokens), nll)


def test_qwen_forward_backward_causality_and_cached_conditionals(arrays):
    model = create_model(ModelConfig()).eval()
    assert sum(p.numel() for p in model.parameters()) == 988032
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    assert (
        model.config.eos_token_id is None
        and model.config.sliding_window is None
        and not model.config.use_sliding_window
    )
    tokens = torch.tensor(arrays["tokens"][:2].astype(np.int64))
    nll = teacher_forced_nll(model, tokens)
    nll.sum(1).mean().backward()
    assert torch.isfinite(model.model.embed_tokens.weight.grad).all()
    first, second = conditional_log_probs(model, tokens[:, :PROMPT_LENGTH])
    joint = joint_log_probs(first, second)
    rows = torch.arange(len(tokens))
    torch.testing.assert_close(-first[rows, tokens[:, -2]], nll[:, 0].double(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(-second[rows, tokens[:, -2], tokens[:, -1]], nll[:, 1].double(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(joint.exp().sum(-1), torch.ones(2, dtype=torch.float64), rtol=0, atol=1e-6)
    modified = tokens.clone()
    modified[:, -1] = (modified[:, -1] + 1) % 16
    with torch.no_grad():
        original_logits = model(tokens, use_cache=False).logits
        modified_logits = model(modified, use_cache=False).logits
    torch.testing.assert_close(original_logits[:, :204], modified_logits[:, :204], rtol=0, atol=0)
    with pytest.raises(ValueError, match="203-token"):
        conditional_log_probs(model, tokens)


def test_exact_estimator_boundaries_and_sampling():
    np.testing.assert_allclose(exact_pass(np.array([0.0, 1 / 256, 1.0]), 1), [0, 1 / 256, 1])
    probabilities = np.array([0.001, 0.02, 0.5])
    counts = np.arange(257)
    for k in (1, 2, 16, 128, 256):
        estimates = estimated_pass(counts, 256, k)
        expected = binom.pmf(counts[None, :], 256, probabilities[:, None]) @ estimates
        np.testing.assert_allclose(expected, exact_pass(probabilities, k), atol=2e-13)
        assert estimates[0] == 0 and estimates[-1] == 1
        for c in (1, 50, 255):
            denominator = math.comb(256, k)
            reference = 1 - (math.comb(256 - c, k) / denominator if 256 - c >= k else 0)
            assert estimates[c] == pytest.approx(reference)
    assert not np.isclose(exact_pass(probabilities, 16).mean(), exact_pass(probabilities.mean(), 16))
    generator = torch.Generator().manual_seed(31)
    first = torch.log_softmax(torch.randn(2, 16, dtype=torch.float64), -1)
    second = torch.log_softmax(torch.randn(2, 16, 16, dtype=torch.float64), -1)
    answers = sample_answers(first, second, 100000, generator).numpy()
    joint = joint_log_probs(first, second).exp().numpy()
    for i in range(2):
        histogram = np.bincount(answers[i, :, 0] * 16 + answers[i, :, 1], minlength=256) / 100000
        np.testing.assert_allclose(histogram, joint[i], atol=0.003)
    result = sampled_summary(np.array([0, 256]), np.array([0.0, 1.0]), 256)
    assert result["generative_pass"]["256"]["mean"] == 0.5
    assert result["generative_pass"]["256"]["conditional_sampling_sd_of_mean"] == 0


def test_reference_normalization_and_analytic_cases(arrays):
    mean, variance = bayesian_predictive(np.zeros((2, 16, 4)), np.zeros((2, 16)), np.ones((2, 4)), 0.5)
    np.testing.assert_allclose(mean, 0)
    np.testing.assert_allclose(variance, 1.25)
    gaussian = gaussian_bin_log_probs(np.array([-100.0, 0.0, 100.0]), np.array([0.25, 0.25, 0.25]))
    assert np.isfinite(gaussian).all()
    np.testing.assert_allclose(np.exp(gaussian).sum(1), 1, atol=1e-13)
    assert np.exp(gaussian[0, 0]) == 1 and np.exp(gaussian[-1, -1]) == 1
    for name, log_probs in reference_distributions(arrays, DatasetConfig()).items():
        report = distribution_summary(log_probs, arrays)
        assert report["max_normalization_error"] < 1e-12
        if name == "uniform_256":
            assert report["answer_nll"]["mean"] == pytest.approx(math.log(256))
            assert report["exact_pass"]["1"]["mean"] == pytest.approx(1 / 256)
            assert report["clean_answer_nll"]["mean"] == pytest.approx(math.log(256))
            assert report["clean_exact_pass"]["1"]["mean"] == pytest.approx(1 / 256)


def test_clean_and_noisy_metrics_use_exact_distribution_and_distinct_targets(arrays):
    selected = {name: values[:2].copy() for name, values in arrays.items()}
    selected["query_signal"] = np.array([1.2, -6.0])
    selected["query_y"] = np.array([1.3, 6.0])
    selected["tokens"][:, -2:] = encode(selected["query_y"])
    clean_indices = quantize(selected["query_signal"])
    noisy_indices = quantize(selected["query_y"])
    clean_p = np.array([0.2, 0.6])
    noisy_p = np.array([0.5, 0.1])
    probabilities = np.full((2, 256), 0.3 / 254)
    probabilities[np.arange(2), clean_indices] = clean_p
    probabilities[np.arange(2), noisy_indices] = noisy_p
    report = distribution_summary(np.log(probabilities), selected)
    # This event deliberately has no samples, train subset, or progress counters.
    values = event_metrics(
        {
            "kind": "evaluation",
            "step": 0,
            "elapsed_seconds": 2.0,
            "evaluation_seconds": 1.0,
            "eval": report,
        }
    )
    assert values["eval/nll/clean"] == pytest.approx(-np.log(clean_p).mean())
    assert values["eval/nll/noisy"] == pytest.approx(-np.log(noisy_p).mean())
    for target, p in (("clean", clean_p), ("noisy", noisy_p)):
        for k in (1, 4, 16, 64, 256):
            assert values[f"pass@k_exact/pass@{k}/{target}"] == pytest.approx((1 - (1 - p) ** k).mean())
    # The background centers plus the two distinguished centers give the exact mean.
    means = 0.3 / 254 * (CENTERS.sum() - CENTERS[clean_indices] - CENTERS[noisy_indices])
    means += clean_p * CENTERS[clean_indices] + noisy_p * CENTERS[noisy_indices]
    for target, continuous in (("clean", selected["query_signal"]), ("noisy", selected["query_y"])):
        mse = ((means - continuous) ** 2).mean()
        assert values[f"eval/mse/{target}"] == pytest.approx(mse)
        assert not np.isclose(mse, ((means - decode(encode(continuous))) ** 2).mean())


def test_sampled_mean_mse_averages_predictions_before_squaring():
    completions = np.full((2, 256, 2), 15, dtype=np.uint8)
    completions[0, :128] = 0  # Half -5, half +5: mean 0, despite sample variance 25.
    signals = np.array([1.0, 2.0])  # Continuous signals, without quantization or query noise.
    result = sampled_mean_mse(completions, signals)
    # Per-prompt squared errors: (0-1)^2 = 1 and (5-2)^2 = 9.
    assert result["mean"] == pytest.approx(5)
    assert result["prompt_se"] == pytest.approx(4)
    assert result["prompts"] == 2
    for samples, targets in (
        (completions[:, :255], signals),
        (completions, signals[:1]),
        (completions, [1.0, np.nan]),
        (completions[:0], signals[:0]),
    ):
        with pytest.raises(ValueError, match="256"):
            sampled_mean_mse(samples, targets)
    completions[0, 0, 0] = 16
    with pytest.raises(ValueError, match="digit IDs"):
        sampled_mean_mse(completions, signals)


def test_checkpoint_resume_reproduces_next_optimizer_step(tmp_path, arrays):
    config = TrainConfig(
        batch_size=4,
        micro_batch_size=2,
        max_steps=3,
        train_eval_size=4,
        device="cpu",
        precision="fp32",
        warmup_steps=2,
    )
    model = create_model(ModelConfig())
    optimizer, _ = make_optimizer(
        model, config.learning_rate, config.beta1, config.beta2, config.weight_decay, config.optimizer_epsilon
    )
    scheduler = make_scheduler(optimizer, config.warmup_steps)
    order = FrozenOrder(8, config.order_seed)
    device = torch.device("cpu")
    first_step = optimize_step(model, optimizer, scheduler, order, arrays["tokens"], config, device)
    assert first_step["learning_rate"] == config.learning_rate / 2
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, optimizer, scheduler, order, 1, config, {}, {}, 0, {})
    expected = optimize_step(model, optimizer, scheduler, order, arrays["tokens"], config, device)
    expected_weights = {name: value.clone() for name, value in model.state_dict().items()}
    load_checkpoint(checkpoint, model, optimizer, scheduler, order, config, {})
    actual = optimize_step(model, optimizer, scheduler, order, arrays["tokens"], config, device)
    assert actual == expected
    assert order.presentations == 8
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(checkpoint, model, optimizer, scheduler, order, replace(config, learning_rate=1e-3), {})


def test_cpu_end_to_end_frozen_evaluation_and_artifacts(tmp_path, monkeypatch):
    def reject_sampling(*args, **kwargs):
        raise AssertionError("Exact evaluation must not sample completions")

    monkeypatch.setattr(torch, "multinomial", reject_sampling)
    pool = tmp_path / "data"
    prepare(pool, DatasetConfig(train_count=8, eval_count=4))
    config = TrainConfig(
        batch_size=4,
        micro_batch_size=2,
        max_steps=2,
        eval_interval=1,
        train_eval_size=4,
        eval_batch_size=2,
        device="cpu",
        precision="fp32",
        cpu_threads=1,
    )
    summary = train(pool, tmp_path / "run", config, ModelConfig())
    assert summary["steps"] == 2 and summary["presentations"] == 8
    assert summary["parameter_count"] == 988032
    events = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    evaluations = [event for event in events if event["kind"] == "evaluation"]
    assert [event["step"] for event in evaluations] == [0, 1, 2]
    assert [event["eval"]["prompts"] for event in evaluations] == [4, 4, 4]
    pools, _ = load_pool(pool)
    for event in evaluations:
        assert "generation" not in event["eval"]
        for metric_group in (event["train"], event["eval"]):
            assert "first_token_nll" not in metric_group
            assert "second_token_conditional_nll" not in metric_group
        with np.load(tmp_path / "run" / f"evaluation-{event['step']:05d}.npz") as archive:
            assert set(archive.files) == {"ids", "log_probs"}
            np.testing.assert_array_equal(archive["ids"], pools["eval"]["ids"])
            per_prompt_errors = (np.exp(archive["log_probs"]) @ CENTERS - pools["eval"]["query_signal"]) ** 2
        mse = event["eval"]["predictive_mean_errors"]["continuous_noiseless_signal"]["mse"]
        assert mse["mean"] == pytest.approx(per_prompt_errors.mean())
        assert mse["prompts"] == 4
    before = array_hash(pools["eval"])
    model = create_model(ModelConfig()).eval()
    rng_before = torch.get_rng_state().clone()
    for suffix in ("one", "two"):
        report = evaluate(model, pools["eval"], 2, torch.device("cpu"), "fp32", tmp_path / f"{suffix}.npz")
        assert "generation" not in report
    with np.load(tmp_path / "one.npz") as one, np.load(tmp_path / "two.npz") as two:
        assert set(one.files) == set(two.files) == {"ids", "log_probs"}
        np.testing.assert_array_equal(one["log_probs"], two["log_probs"])
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)
    assert array_hash(pools["eval"]) == before
    with pytest.raises(ValueError, match="positive evaluation batch size"):
        evaluate(model, pools["eval"], 0, torch.device("cpu"), "fp32")
    # The launcher must still finish its report after the generation block is removed.
    from noisy_regression.report import render_report

    render_report(tmp_path / "run")
    rendered = (tmp_path / "run" / "report.md").read_text()
    assert "No completions were sampled" in rendered
    assert "Clean MSE" in rendered and "Clean exact pass@k" in rendered
    assert "Conditional sampling" not in rendered
    for filename in ("learning_curves.png", "pass_at_k.png"):
        assert (tmp_path / "run" / filename).stat().st_size > 0


@pytest.fixture
def recorded_wandb(monkeypatch):
    class RecordingRun:
        id = "cpu-validation"
        url = "https://wandb.invalid/cpu-validation"

        def __init__(self):
            self.history = []
            self.summary = {}
            self.finished = False

        def log(self, metrics, step):
            self.history.append((step, metrics.copy()))

        def finish(self):
            self.finished = True

    recorded_run = RecordingRun()
    init_arguments = {}

    def initialize(**kwargs):
        init_arguments.update(kwargs)
        return recorded_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    return recorded_run, init_arguments


def test_wandb_combines_same_step_metrics_without_accumulation(tmp_path, recorded_wandb):
    recorded_run, init_arguments = recorded_wandb
    pool = tmp_path / "data"
    prepare(pool, DatasetConfig(train_count=8, eval_count=4, sigma=0.1))
    assert TrainConfig().batch_size == TrainConfig().micro_batch_size == 64
    config = TrainConfig(
        batch_size=4,
        micro_batch_size=4,
        max_steps=2,
        eval_interval=1,
        train_eval_size=4,
        eval_batch_size=2,
        device="cpu",
        precision="fp32",
        cpu_threads=1,
        log_interval=1,
    )
    tracking = TrackingConfig(True, "noisy-regression-sft", "cpu-validation")
    summary = train(pool, tmp_path / "run", config, ModelConfig(), tracking_config=tracking)
    assert summary["presentations"] == 8
    assert init_arguments["mode"] == "online" and init_arguments["project"] == tracking.project_name
    assert init_arguments["config"]["dataset"]["sigma"] == 0.1
    assert init_arguments["config"]["micro_batch_size"] == init_arguments["config"]["batch_size"] == 4
    assert init_arguments["config"]["dashboard_schema_version"] == 3
    assert init_arguments["config"]["dashboard_pass_k"] == [1, 4, 16, 64, 256]
    assert init_arguments["config"]["evaluation_prompts"] == 4
    assert init_arguments["config"]["method"] == "sft"
    assert [step for step, _ in recorded_run.history] == [0, 1, 2]
    events = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    evaluations = {event["step"]: event for event in events if event["kind"] == "evaluation"}
    optimizations = {event["step"]: event for event in events if event["kind"] == "optimization"}
    references = json.loads((tmp_path / "run" / "references.json").read_text())
    for step, values in recorded_run.history:
        evaluation = evaluations[step]["eval"]
        assert values["eval/nll/noisy"] == evaluation["answer_nll"]["mean"]
        assert values["eval/nll/clean"] == evaluation["clean_answer_nll"]["mean"]
        assert (
            values["eval/mse/clean"]
            == evaluation["predictive_mean_errors"]["continuous_noiseless_signal"]["mse"]["mean"]
        )
        assert (
            values["eval/mse/noisy"] == evaluation["predictive_mean_errors"]["continuous_noisy_outcome"]["mse"]["mean"]
        )
        assert {key for key in values if key.startswith("pass")} == {
            f"pass@k_exact/pass@{k}/{target}" for target in ("clean", "noisy") for k in (1, 4, 16, 64, 256)
        }
        for target, source in (("clean", "clean_exact_pass"), ("noisy", "exact_pass")):
            for k in (1, 4, 16, 64, 256):
                assert values[f"pass@k_exact/pass@{k}/{target}"] == evaluation[source][str(k)]["mean"]
        assert values["timing/evaluation_seconds"] == evaluations[step]["evaluation_seconds"]
        assert 0 < values["timing/evaluation_seconds"] <= values["timing/elapsed_seconds"]
        assert not any(
            unwanted in key
            for key in values
            for unwanted in ("prompts", "prompt_se", "normal95", "sampled", "sampling_sd", "example_ids")
        )
        assert not any(
            key.startswith(("reference/", "train_eval/", "trainer/", "progress/", "regression/", "likelihood/"))
            for key in values
        )
        assert len(values) == (17 if step == 0 else 22 if step == 2 else 21)
        if step == 2:
            assert values["diagnostics/context_shuffle_nll_increase"] == pytest.approx(
                evaluation["mismatched_context_control"]["answer_nll"]["mean"] - evaluation["answer_nll"]["mean"]
            )
        else:
            assert "diagnostics/context_shuffle_nll_increase" not in values
        if step:
            assert "train/answer_nll" in values and "train/learning_rate" in values
            norm = values["train/gradient_norm_before_clip"]
            assert norm == optimizations[step]["gradient_norm_before_clip"]
            assert math.isfinite(norm) and norm > 0
            assert 0 < values["timing/optimizer_step_seconds"] < values["timing/elapsed_seconds"]
    # New numeric artifact diagnostics must never silently become dashboard panels.
    final_event = evaluations[2]
    curated = event_metrics(final_event)
    final_event["eval"]["future_diagnostic"] = {"mean": 123, "prompts": 4}
    del final_event["train"]
    assert "generation" not in final_event["eval"]
    assert event_metrics(final_event) == curated
    assert "prompt_se" in final_event["eval"]["answer_nll"]
    assert len(final_event["eval"]["exact_pass"]) == 9
    assert len(final_event["eval"]["clean_exact_pass"]) == 9
    assert len(references) == 5
    assert recorded_run.finished
    assert recorded_run.summary == {
        "result/best_answer_nll": summary["best"]["answer_nll"],
        "result/best_step": summary["best"]["step"],
        "result/best_checkpoint": summary["best"]["checkpoint"],
        "result/final_checkpoint": summary["final_checkpoint"],
    }
    assert json.loads((tmp_path / "run" / "wandb_run.json").read_text())["id"] == recorded_run.id


@pytest.mark.parametrize(
    "method,reference",
    [
        ("bayesian", "bayesian_continuous_optimistic"),
        ("ridge", "ridge_decoded_gaussian_approximation"),
    ],
)
def test_baseline_logs_same_exact_metrics_once_at_step_zero(tmp_path, recorded_wandb, method, reference):
    recorded_run, init_arguments = recorded_wandb
    pool = tmp_path / "data"
    dataset_config = DatasetConfig(train_count=8, eval_count=4, sigma=0.1)
    prepare(pool, dataset_config)
    tracking = TrackingConfig(True, "noisy-regression-sft", f"{method}-validation")
    output = tmp_path / method
    event = evaluate_baseline(pool, output, method, tracking)
    assert [step for step, _ in recorded_run.history] == [0]
    values = recorded_run.history[0][1]
    assert values == event_metrics(event)
    assert len(values) == 17
    assert recorded_run.finished
    assert init_arguments["project"] == tracking.project_name
    assert init_arguments["config"]["method"] == method
    assert init_arguments["config"]["reference_distribution"] == reference
    assert init_arguments["config"]["evaluation_prompts"] == 4
    assert not any(key.startswith(("reference/", "train/", "progress/")) for key in values)
    assert "generation" not in event["eval"]
    pools, _ = load_pool(pool)
    expected = reference_distributions(pools["eval"], dataset_config)[reference]
    with np.load(output / "per_prompt.npz") as archive:
        np.testing.assert_allclose(archive["log_probs"], expected, atol=1e-13)
        np.testing.assert_array_equal(archive["ids"], pools["eval"]["ids"])
        assert set(archive.files) == {"ids", "log_probs"}
    assert json.loads((output / "metrics.json").read_text()) == event
    with pytest.raises(FileExistsError):
        evaluate_baseline(pool, output, method, tracking)
    assert len(recorded_run.history) == 1
