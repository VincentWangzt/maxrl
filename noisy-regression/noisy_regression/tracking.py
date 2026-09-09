"""Curated W&B curves; full evaluation diagnostics stay in JSON/NPZ artifacts."""

from dataclasses import dataclass

from noisy_regression.data import write_json

DASHBOARD_KS = (1, 4, 16, 64, 256)


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    project_name: str
    experiment_name: str


def evaluation_metrics(evaluation):
    """Identical full-pool metric names for learned methods and baseline runs."""
    metrics = {
        "eval/mse/clean": evaluation["predictive_mean_errors"]["continuous_noiseless_signal"]["mse"]["mean"],
        "eval/mse/noisy": evaluation["predictive_mean_errors"]["continuous_noisy_outcome"]["mse"]["mean"],
        "eval/nll/clean": evaluation["clean_answer_nll"]["mean"],
        "eval/nll/noisy": evaluation["answer_nll"]["mean"],
        "diagnostics/predictive_entropy_nats": evaluation["entropy_nats_per_answer"]["mean"],
    }
    for target, source in (("clean", "clean_exact_pass"), ("noisy", "exact_pass")):
        for k in DASHBOARD_KS:
            metrics[f"pass@k_exact/pass@{k}/{target}"] = evaluation[source][str(k)]["mean"]
    if "mismatched_context_control" in evaluation:
        metrics["diagnostics/context_shuffle_nll_increase"] = (
            evaluation["mismatched_context_control"]["answer_nll"]["mean"] - evaluation["answer_nll"]["mean"]
        )
    return metrics


def event_metrics(event):
    metrics = {"timing/elapsed_seconds": event["elapsed_seconds"]}
    if event["kind"] == "optimization":
        metrics.update(
            {
                "train/answer_nll": event["answer_nll"],
                "train/gradient_norm_before_clip": event["gradient_norm_before_clip"],
                "train/learning_rate": event["learning_rate"],
                "timing/optimizer_step_seconds": event["optimizer_step_seconds"],
            }
        )
    elif event["kind"] == "evaluation":
        metrics.update(evaluation_metrics(event["eval"]))
        if "train" in event:
            train_errors = event["train"]["predictive_mean_errors"]
            metrics.update(
                {
                    "train_probe/mse/clean": train_errors["continuous_noiseless_signal"]["mse"]["mean"],
                    "train_probe/mse/noisy": train_errors["continuous_noisy_outcome"]["mse"]["mean"],
                }
            )
        metrics["timing/evaluation_seconds"] = event["evaluation_seconds"]
    else:
        raise ValueError(f"Unknown metric event kind: {event['kind']}")
    return metrics


class WandbLogger:
    def __init__(self, run):
        self.run = run
        self.pending_step = None
        self.pending_metrics = {}

    def record(self, event):
        step = event["step"]
        if self.pending_step is not None and step < self.pending_step:
            raise ValueError("W&B events must have nondecreasing optimizer steps")
        if self.pending_step is not None and step > self.pending_step:
            self.flush()
        self.pending_step = step
        self.pending_metrics.update(event_metrics(event))

    def flush(self):
        if self.pending_step is not None:
            self.run.log(self.pending_metrics, step=self.pending_step)
            self.pending_step = None
            self.pending_metrics = {}

    def finish(self, summary):
        self.flush()
        self.run.summary.update(summary)
        self.run.finish()


def initialize_tracking(config, output_path, run_config, metadata):
    if not config.enabled:
        return None
    if not config.project_name or not config.experiment_name:
        raise ValueError("W&B project and experiment names must be explicit")
    # Explicit failure if W&B is requested but unavailable; no silent fallback.
    import wandb

    run = wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        dir=str(output_path),
        mode="online",
        config={
            **run_config,
            "dataset": metadata["config"],
            "dataset_hashes": {split: item["content_sha256"] for split, item in metadata["splits"].items()},
            "dashboard_schema_version": 4,
            "dashboard_pass_k": list(DASHBOARD_KS),
            "evaluation_prompts": metadata["config"]["eval_count"],
            "metric_targets": {
                "mse": "Exact grid-distribution mean vs continuous signal (clean) or outcome (noisy)",
                "nll_and_pass": "Same distribution scored at quantized signal (clean) or outcome (noisy)",
            },
        },
    )
    write_json(
        output_path / "wandb_run.json",
        {"id": run.id, "url": run.url, "project": config.project_name, "name": config.experiment_name},
    )
    print(f"W&B run: {run.url}", flush=True)
    return WandbLogger(run)
