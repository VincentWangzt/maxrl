"""Evaluate a saved model on the frozen held-out pool."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from noisy_regression.codec import DIGITS, PROMPT_LENGTH
from noisy_regression.data import fixed_subset_indices, load_pool, subset, write_json
from noisy_regression.metrics import distribution_summary, mean_se, sampled_mean_mse, sampled_summary
from noisy_regression.model import conditional_log_probs, joint_log_probs, sample_answers, teacher_forced_nll


def precision_context(device, precision):
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 experiment execution requires the explicitly selected CUDA GPU")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision != "fp32":
        raise ValueError("Precision must be fp32 or bf16")
    return nullcontext()


def select_device(name, precision):
    if name not in ("cpu", "cuda:0"):
        raise ValueError("Use cpu or cuda:0 with CUDA_VISIBLE_DEVICES set to the explicitly selected physical GPU")
    device = torch.device(name)
    if device.type == "cuda":
        import os

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible or torch.cuda.device_count() != 1:
            raise ValueError("CUDA_VISIBLE_DEVICES must identify exactly one explicitly selected GPU")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected GPU does not support BF16")
    elif precision != "fp32":
        raise ValueError("CPU validation requires fp32")
    return device


@torch.inference_mode()
def likelihood(model, tokens, batch_size, device, precision):
    model.eval()
    losses = []
    for start in range(0, len(tokens), batch_size):
        batch = torch.tensor(tokens[start : start + batch_size].astype(np.int64), device=device)
        with precision_context(device, precision):
            losses.append(teacher_forced_nll(model, batch).double().cpu().numpy())
    losses = np.concatenate(losses)
    return {
        "answer_nll": mean_se(losses.sum(1)),
        "answer_log_likelihood": mean_se(-losses.sum(1)),
    }


@torch.inference_mode()
def evaluate(
    model,
    arrays,
    generation_indices,
    eval_batch_size,
    generation_batch_size,
    samples,
    sampling_seed,
    device,
    precision,
    artifact_path=None,
):
    if min(eval_batch_size, generation_batch_size) < 1 or samples != 256:
        raise ValueError("Require positive batch sizes and exactly 256 completions per prompt")
    indices = np.asarray(generation_indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or len(indices) < 1
        or len(np.unique(indices)) != len(indices)
        or np.any((indices < 0) | (indices >= len(arrays["tokens"])))
    ):
        raise ValueError("Generation subset must contain unique valid example indices")
    model.eval()
    first_parts, second_parts = [], []
    for start in range(0, len(arrays["tokens"]), eval_batch_size):
        prompt = torch.tensor(
            arrays["tokens"][start : start + eval_batch_size, :PROMPT_LENGTH].astype(np.int64), device=device
        )
        with precision_context(device, precision):
            first, second = conditional_log_probs(model, prompt)
        joint_log_probs(first, second)
        first_parts.append(first.cpu())
        second_parts.append(second.cpu())
    first, second = torch.cat(first_parts), torch.cat(second_parts)
    log_probs = joint_log_probs(first, second).numpy()
    report = distribution_summary(log_probs, arrays)
    generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
    completions = []
    for start in range(0, len(indices), generation_batch_size):
        selected = indices[start : start + generation_batch_size]
        completions.append(sample_answers(first[selected], second[selected], samples, generator).numpy())
    completions = np.concatenate(completions)
    targets = arrays["tokens"][indices, -2:].astype(np.int64)
    valid = ((completions >= 0) & (completions < DIGITS)).all(-1)
    successes = (valid & (completions == targets[:, None, :]).all(-1)).sum(1)
    target_indices = targets[:, 0] * 16 + targets[:, 1]
    probabilities = np.exp(log_probs[indices, target_indices])
    report["generation"] = sampled_summary(successes, probabilities, samples)
    report["generation"].update(
        {
            "sampled_mean_mse": sampled_mean_mse(completions, arrays["query_signal"][indices]),
            "invalid_completions": int((~valid).sum()),
            "overlength_completions": 0,
            "output_length_tokens": 2,
            "temperature": 1.0,
            "top_k": None,
            "top_p": None,
            "sampling_seed": sampling_seed,
            "generation_batch_size_prompts": generation_batch_size,
            "eval_batch_size_prompts": eval_batch_size,
            "sampling_device": "cpu-float64 from model conditional probabilities",
            "example_ids": arrays["ids"][indices].tolist(),
        }
    )
    if artifact_path is not None:
        np.savez_compressed(
            artifact_path,
            ids=arrays["ids"],
            log_probs=log_probs,
            generation_indices=indices,
            generation_ids=arrays["ids"][indices],
            completions=completions.astype(np.uint8),
            success_counts=successes,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    parser.add_argument("--precision", choices=["fp32", "bf16"], required=True)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--generation-subset-size", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--subset-seed", type=int, default=5772)
    parser.add_argument("--sampling-seed", type=int, default=8119)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    device = select_device(args.device, args.precision)
    splits, metadata = load_pool(args.data)
    saved_metadata = json.loads((args.checkpoint / "dataset_metadata.json").read_text())
    if metadata != saved_metadata:
        raise ValueError("Checkpoint and dataset metadata differ")
    if not 1 <= args.generation_subset_size <= len(splits["eval"]["tokens"]):
        raise ValueError("Generation subset exceeds held-out pool")
    args.output.mkdir(parents=True, exist_ok=False)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, attn_implementation="sdpa", local_files_only=True).to(
        device
    )
    _, indices = fixed_subset_indices(
        len(splits["train"]["tokens"]), len(splits["eval"]["tokens"]), 0, args.generation_subset_size, args.subset_seed
    )
    if args.generation_subset_size == len(splits["eval"]["tokens"]):
        indices = np.arange(len(splits["eval"]["tokens"]))
    report = evaluate(
        model,
        splits["eval"],
        indices,
        args.eval_batch_size,
        args.generation_batch_size,
        args.samples,
        args.sampling_seed,
        device,
        args.precision,
        args.output / "per_prompt.npz",
    )
    # A context mismatch control preserves every query and its stored target.
    shuffled = subset(splits["eval"], np.arange(len(splits["eval"]["tokens"])))
    shuffled["tokens"][:, 1:193] = np.roll(shuffled["tokens"][:, 1:193], 1, axis=0)
    report["mismatched_context_control"] = likelihood(
        model, shuffled["tokens"], args.eval_batch_size, device, args.precision
    )
    report["checkpoint"] = str(args.checkpoint.resolve())
    write_json(args.output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
