"""Resumable fixed-pool SFT; all execution belongs on cmu-L40-live."""

import argparse
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
import torch
import transformers
from transformers import AutoModelForCausalLM

from noisy_regression.codec import CONTEXT_SLICE, save_codec
from noisy_regression.data import DatasetConfig, FrozenOrder, load_pool, subset, write_json
from noisy_regression.evaluate import evaluate, likelihood, precision_context, select_device
from noisy_regression.model import ModelConfig, create_model, make_optimizer, make_scheduler, teacher_forced_nll
from noisy_regression.references import reference_report
from noisy_regression.tracking import TrackingConfig, initialize_tracking


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    micro_batch_size: int = 128
    max_steps: int = 80_000
    eval_interval: int = 500
    learning_rate: float = 1e-4
    min_learning_rate: float = 0.0
    learning_rate_schedule: str = "linear_warmup_constant"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    optimizer_epsilon: float = 1e-8
    warmup_steps: int = 1_600
    max_grad_norm: float | None = None
    eval_batch_size: int = 32
    device: str = "cuda:0"
    precision: str = "bf16"
    cpu_threads: int = 4
    log_interval: int = 10

    def validate(self, splits):
        integers = (
            self.batch_size,
            self.micro_batch_size,
            self.max_steps,
            self.eval_interval,
            self.eval_batch_size,
            self.cpu_threads,
            self.log_interval,
        )
        if min(integers) < 1 or self.batch_size % self.micro_batch_size:
            raise ValueError("Positive sizes required; effective batch must be divisible by microbatch")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("Require 0 <= warmup_steps < max_steps")
        if self.learning_rate_schedule not in ("linear_warmup_cosine_decay", "linear_warmup_constant"):
            raise ValueError("Unknown learning-rate schedule")
        if (
            not 0 <= self.beta1 < 1
            or not 0 <= self.beta2 < 1
            or min(self.learning_rate, self.optimizer_epsilon) <= 0
            or not 0 <= self.min_learning_rate <= self.learning_rate
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid optimizer settings")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be None or a finite positive value")


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scheduler, order, step, config, metadata, best, elapsed, metrics):
    path = Path(path)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(temporary)
    save_codec(temporary)
    write_json(temporary / "training_config.json", asdict(config))
    write_json(temporary / "dataset_metadata.json", metadata)
    write_json(temporary / "metrics.json", metrics)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "order": order.state_dict(),
            "step": step,
            "rng": rng_state(),
            "best": best,
            "elapsed_seconds": elapsed,
        },
        temporary / "trainer_state.pt",
    )
    temporary.rename(path)


def load_checkpoint(path, model, optimizer, scheduler, order, config, metadata):
    path = Path(path)
    if json.loads((path / "training_config.json").read_text()) != asdict(config):
        raise ValueError("Resume requires the same complete training configuration")
    if json.loads((path / "dataset_metadata.json").read_text()) != metadata:
        raise ValueError("Resume dataset fingerprint/configuration mismatch")
    saved_model_config = json.loads((path / "config.json").read_text())
    for name in ModelConfig.__dataclass_fields__:
        if saved_model_config.get(name) != getattr(model.config, name):
            raise ValueError(f"Resume architecture mismatch: {name}")
    restored = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, attn_implementation="sdpa")
    model.load_state_dict(restored.state_dict())
    # Only load trusted checkpoints produced by this experiment; optimizer/RNG
    # state includes Python and NumPy objects, not just tensors.
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    order.load_state_dict(state["order"])
    restore_rng(state["rng"])
    if order.presentations != state["step"] * config.batch_size:
        raise ValueError("Checkpoint example-presentation counter mismatch")
    return state


def optimize_step(model, optimizer, scheduler, order, train_tokens, config, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    indices = order.take(config.batch_size)
    loss_sum = torch.zeros((), device=device)
    for start in range(0, config.batch_size, config.micro_batch_size):
        tokens = torch.tensor(
            train_tokens[indices[start : start + config.micro_batch_size]].astype(np.int64), device=device
        )
        with precision_context(device, config.precision):
            nll = teacher_forced_nll(model, tokens).sum(1)
            loss = nll.sum() / config.batch_size
        loss.backward()
        loss_sum += nll.detach().sum()
    if config.max_grad_norm is None:
        gradient_norm = torch.nn.utils.get_total_norm(
            (parameter.grad for parameter in model.parameters() if parameter.grad is not None),
            norm_type=2.0,
            error_if_nonfinite=True,
        )
    else:
        # Report the global norm before clipping, after all microbatches accumulate.
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm, norm_type=2.0, error_if_nonfinite=True
        )
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    return (
        {
            "answer_nll": loss_sum.item() / config.batch_size,
            "gradient_norm": float(gradient_norm),
            "learning_rate": lr,
        },
        indices,
    )


def train(data_path, output_path, config, model_config, resume=None, tracking_config=None):
    started = time.perf_counter()
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = select_device(config.device, config.precision)
    print(f"Loading and verifying frozen pool: {data_path}", flush=True)
    splits, metadata = load_pool(data_path)
    config.validate(splits)
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    model = create_model(model_config).to(device)
    optimizer, decay_groups = make_optimizer(
        model, config.learning_rate, config.beta1, config.beta2, config.weight_decay, config.optimizer_epsilon
    )
    scheduler = make_scheduler(
        optimizer,
        config.warmup_steps,
        config.max_steps,
        config.min_learning_rate,
        config.learning_rate_schedule,
    )
    order = FrozenOrder(len(splits["train"]["tokens"]))
    step, previous_elapsed = 0, 0.0
    best = {"answer_nll": None, "step": None, "checkpoint": None}
    if resume is not None:
        state = load_checkpoint(resume, model, optimizer, scheduler, order, config, metadata)
        step, previous_elapsed, best = state["step"], state["elapsed_seconds"], state["best"]
        if step >= config.max_steps:
            raise ValueError("Checkpoint has already finished the requested optimizer steps")
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda": torch.version.cuda,
    }
    environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG",
            "PYTORCH_CUDA_ALLOC_CONF",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    }
    manifest = {
        "training": asdict(config),
        "architecture": asdict(model_config),
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer_parameter_groups": decay_groups,
        "optimizer": "AdamW; FP32 parameters/states, foreach=False, fused=False",
        "learning_rate_schedule": (
            f"{config.learning_rate_schedule}; linear warmup from {config.min_learning_rate:g} to "
            f"{config.learning_rate:g}, then "
            + (
                f"cosine decay to {config.min_learning_rate:g}"
                if config.learning_rate_schedule == "linear_warmup_cosine_decay"
                else "constant base rate"
            )
        ),
        "versions": versions,
        "environment": environment,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_path": str(Path(data_path).resolve()),
        "resume_from": str(Path(resume).resolve()) if resume else None,
        "deterministic_algorithms": True,
        "randomness": "Fresh initialization and training shuffle; no fixed seeds",
        "likelihood_units": "nats per complete two-token answer",
        "evaluation_policy": "Full held-out pool plus the just-optimized training batch at evaluation steps",
    }
    write_json(output_path / "manifest.json", manifest)
    write_json(output_path / "dataset_metadata.json", metadata)
    references = reference_report(splits["eval"], DatasetConfig(**metadata["config"]))
    write_json(output_path / "references.json", references)
    save_codec(output_path)
    metrics_path = output_path / "metrics.jsonl"
    tracker = None
    if tracking_config is not None:
        state_before_tracking = rng_state()
        tracker = initialize_tracking(
            tracking_config,
            output_path,
            {
                **manifest["training"],
                "method": "sft",
                **{
                    name: manifest[name]
                    for name in (
                        "architecture",
                        "parameter_count",
                        "optimizer_parameter_groups",
                        "versions",
                        "git_commit",
                        "resume_from",
                    )
                },
            },
            metadata,
        )
        restore_rng(state_before_tracking)

    def emit(event):
        event["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if tracker is not None:
            tracker.record(event)
        if event["kind"] == "evaluation":
            printed = {
                "step": event["step"],
                "held_out_nll": event["eval"]["answer_nll"]["mean"],
                "exact_pass1": event["eval"]["exact_pass"]["1"]["mean"],
                "elapsed_seconds": event["elapsed_seconds"],
            }
            if "train_batch" in event:
                printed["train_batch_clean_mse"] = event["train_batch"]["predictive_mean_errors"][
                    "continuous_noiseless_signal"
                ]["mse"]["mean"]
            print(json.dumps(printed), flush=True)
        else:
            print(json.dumps(event), flush=True)

    def evaluate_and_save(current_step, train_batch_indices=None):
        nonlocal best
        evaluation_started = time.perf_counter()
        checkpoint = output_path / f"checkpoint-{current_step:05d}"
        final = current_step == config.max_steps
        metrics = {
            "kind": "evaluation",
            "step": current_step,
            "presentations": order.presentations,
            "eval": evaluate(
                model,
                splits["eval"],
                config.eval_batch_size,
                device,
                config.precision,
                output_path / f"evaluation-{current_step:05d}.npz",
            ),
        }
        if train_batch_indices is not None:
            train_batch = subset(splits["train"], train_batch_indices)
            metrics["train_batch"] = evaluate(
                model,
                train_batch,
                config.eval_batch_size,
                device,
                config.precision,
            )
            metrics["train_batch_ids"] = train_batch["ids"].tolist()
        if final:
            control_tokens = splits["eval"]["tokens"].copy()
            control_tokens[:, CONTEXT_SLICE] = np.roll(control_tokens[:, CONTEXT_SLICE], 1, axis=0)
            metrics["eval"]["mismatched_context_control"] = likelihood(
                model, control_tokens, config.eval_batch_size, device, config.precision
            )
        metrics["evaluation_seconds"] = time.perf_counter() - evaluation_started
        nll = metrics["eval"]["answer_nll"]["mean"]
        if best["answer_nll"] is None or nll < best["answer_nll"]:
            best = {"answer_nll": nll, "step": current_step, "checkpoint": str(checkpoint)}
        emit(metrics)
        elapsed = previous_elapsed + time.perf_counter() - started
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            order,
            current_step,
            config,
            metadata,
            best,
            elapsed,
            metrics,
        )
        write_json(output_path / "best_checkpoint.json", best)
        return metrics

    if resume is None:
        evaluate_and_save(0)
    first_step = step + 1
    for step in range(first_step, config.max_steps + 1):
        optimizer_step_started = time.perf_counter()
        step_metrics, train_batch_indices = optimize_step(
            model, optimizer, scheduler, order, splits["train"]["tokens"], config, device
        )
        step_metrics["optimizer_step_seconds"] = time.perf_counter() - optimizer_step_started
        if step % config.log_interval == 0 or step == 1:
            emit({"kind": "optimization", "step": step, "presentations": order.presentations, **step_metrics})
        if step % config.eval_interval == 0 or step == config.max_steps:
            evaluate_and_save(step, train_batch_indices)
    elapsed = previous_elapsed + time.perf_counter() - started
    summary = {
        "final_checkpoint": str(output_path / f"checkpoint-{step:05d}"),
        "best": best,
        "steps": step,
        "presentations": order.presentations,
        "equivalent_passes": order.presentations / len(splits["train"]["tokens"]),
        "elapsed_seconds": elapsed,
        "parameter_count": manifest["parameter_count"],
        "split_role": metadata["split_role"],
    }
    write_json(output_path / "summary.json", summary)
    if tracker is not None:
        tracker.finish(
            {
                "result/best_answer_nll": summary["best"]["answer_nll"],
                "result/best_step": summary["best"]["step"],
                "result/best_checkpoint": summary["best"]["checkpoint"],
                "result/final_checkpoint": summary["final_checkpoint"],
            }
        )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_max_grad_norm(value):
    if value == "none":
        return None
    norm = float(value)
    if not math.isfinite(norm) or norm <= 0:
        raise argparse.ArgumentTypeError("max-grad-norm must be 'none' or a finite positive value")
    return norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--project-name", default="noisy-regression-sft")
    parser.add_argument(
        "--experiment-name",
        default="qwen2_4layer_d2_n64_1m_xy_range3_sft_80000_bs128_lr1e-4_warmup1600_constant_noclip_sigma0p001",
    )
    parser.add_argument(
        "--model-config-json", required=True, help="Complete explicit ModelConfig JSON from the launcher"
    )
    for name, field in TrainConfig.__dataclass_fields__.items():
        argument_type = parse_max_grad_norm if name == "max_grad_norm" else type(field.default)
        parser.add_argument(f"--{name.replace('_', '-')}", type=argument_type, default=field.default)
    args = vars(parser.parse_args())
    data, output, resume = args.pop("data"), args.pop("output"), args.pop("resume")
    model_config = ModelConfig(**json.loads(args.pop("model_config_json")))
    tracking = TrackingConfig(args.pop("use_wandb"), args.pop("project_name"), args.pop("experiment_name"))
    train(data, output, TrainConfig(**args), model_config, resume, tracking)


if __name__ == "__main__":
    main()
