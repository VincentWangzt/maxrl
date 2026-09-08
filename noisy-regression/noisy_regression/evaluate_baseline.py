"""Evaluate one analytical baseline on CPU and optionally log a W&B run at step 0."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from noisy_regression.data import DatasetConfig, load_pool, write_json
from noisy_regression.metrics import distribution_summary
from noisy_regression.references import regression_log_probs
from noisy_regression.tracking import TrackingConfig, initialize_tracking

BASELINE_METHODS = {
    "bayesian": "bayesian_continuous_optimistic",
    "ridge": "ridge_decoded_gaussian_approximation",
}


def evaluate_baseline(data_path, output_path, method, tracking_config):
    if method not in BASELINE_METHODS:
        raise ValueError(f"Unknown baseline method: {method}")
    started = time.perf_counter()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    splits, metadata = load_pool(data_path)
    arrays = splits["eval"]
    evaluation_started = time.perf_counter()
    log_probs = regression_log_probs(arrays, DatasetConfig(**metadata["config"]), decoded=method == "ridge")
    metrics = distribution_summary(log_probs, arrays)
    event = {
        "kind": "evaluation",
        "step": 0,
        "eval": metrics,
        "evaluation_seconds": time.perf_counter() - evaluation_started,
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_config = {
        "method": method,
        "reference_distribution": BASELINE_METHODS[method],
        "input_precision": "continuous" if method == "bayesian" else "quantized",
        "predictive_target": "noisy outcome; the same distribution is scored against clean and noisy targets",
        "data_path": str(Path(data_path).resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": "cpu",
    }
    write_json(output_path / "manifest.json", run_config)
    write_json(output_path / "dataset_metadata.json", metadata)
    write_json(output_path / "metrics.json", event)
    np.savez_compressed(output_path / "per_prompt.npz", ids=arrays["ids"], log_probs=log_probs)
    tracker = initialize_tracking(tracking_config, output_path, run_config, metadata)
    if tracker is not None:
        tracker.record(event)
        tracker.finish({})
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=BASELINE_METHODS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()
    tracking = TrackingConfig(args.use_wandb, args.project_name, args.experiment_name)
    event = evaluate_baseline(args.data, args.output, args.method, tracking)
    print(json.dumps(event, indent=2), flush=True)


if __name__ == "__main__":
    main()
