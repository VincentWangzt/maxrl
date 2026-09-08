"""Evaluate exact answer distributions on the frozen held-out pool; no sampling."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from noisy_regression.codec import PROMPT_LENGTH
from noisy_regression.data import load_pool, subset, write_json
from noisy_regression.metrics import distribution_summary, mean_se
from noisy_regression.model import conditional_log_probs, joint_log_probs, teacher_forced_nll


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
    eval_batch_size,
    device,
    precision,
    artifact_path=None,
):
    if eval_batch_size < 1 or len(arrays["tokens"]) < 1:
        raise ValueError("Require a positive evaluation batch size and a nonempty pool")
    model.eval()
    log_prob_parts = []
    for start in range(0, len(arrays["tokens"]), eval_batch_size):
        prompt = torch.tensor(
            arrays["tokens"][start : start + eval_batch_size, :PROMPT_LENGTH].astype(np.int64), device=device
        )
        with precision_context(device, precision):
            first, second = conditional_log_probs(model, prompt)
        log_prob_parts.append(joint_log_probs(first, second).cpu().numpy())
    log_probs = np.concatenate(log_prob_parts)
    report = distribution_summary(log_probs, arrays)
    if artifact_path is not None:
        np.savez_compressed(
            artifact_path,
            ids=arrays["ids"],
            log_probs=log_probs,
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
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    device = select_device(args.device, args.precision)
    splits, metadata = load_pool(args.data)
    saved_metadata = json.loads((args.checkpoint / "dataset_metadata.json").read_text())
    if metadata != saved_metadata:
        raise ValueError("Checkpoint and dataset metadata differ")
    args.output.mkdir(parents=True, exist_ok=False)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, attn_implementation="sdpa", local_files_only=True).to(
        device
    )
    report = evaluate(
        model,
        splits["eval"],
        args.eval_batch_size,
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
