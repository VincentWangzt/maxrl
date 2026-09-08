"""W&B scalar logging with one committed history row per optimizer step."""

from dataclasses import dataclass
from numbers import Real

from noisy_regression.data import write_json


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    project_name: str
    experiment_name: str


def flatten_metrics(values, prefix=""):
    result = {}
    for name, value in values.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            if name in {"exact_pass", "generative_pass", "exact_pass_same_subset", "sampled_minus_exact"}:
                for k, metrics in value.items():
                    result.update(flatten_metrics(metrics, f"{key}@{k}"))
            else:
                result.update(flatten_metrics(value, key))
        elif isinstance(value, Real) and not isinstance(value, bool):
            result[prefix if name == "mean" else key] = value
        elif name == "prompt_normal95":
            result[f"{prefix}/prompt_normal95_low"], result[f"{prefix}/prompt_normal95_high"] = value
        # IDs, labels, paths, and configuration arrays belong in local artifacts,
        # not scalar histories. In particular, do not log example IDs as metrics.
    return result


def event_metrics(event):
    metrics = {
        "trainer/global_step": event["step"],
        "trainer/presentations": event["presentations"],
        "trainer/elapsed_seconds": event["elapsed_seconds"],
    }
    if event["kind"] == "optimization":
        metrics.update(
            {f"train/{name}": event[name] for name in ("answer_nll", "gradient_norm_before_clip", "learning_rate")}
        )
    elif event["kind"] == "evaluation":
        metrics.update(flatten_metrics({"train_eval": event["train"], "eval": event["eval"]}))
    else:
        raise ValueError(f"Unknown metric event kind: {event['kind']}")
    return metrics


class WandbLogger:
    def __init__(self, run, references):
        self.run = run
        self.references = flatten_metrics({"reference": references})
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
        if event["kind"] == "evaluation":
            self.pending_metrics.update(self.references)

    def flush(self):
        if self.pending_step is not None:
            self.run.log(self.pending_metrics, step=self.pending_step)
            self.pending_step = None
            self.pending_metrics = {}

    def finish(self, summary):
        self.flush()
        self.run.summary.update(flatten_metrics({"result": summary}))
        self.run.finish()


def initialize_tracking(config, output_path, manifest, metadata, references):
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
            **manifest["training"],
            "architecture": manifest["architecture"],
            "parameter_count": manifest["parameter_count"],
            "optimizer_parameter_groups": manifest["optimizer_parameter_groups"],
            "versions": manifest["versions"],
            "git_commit": manifest["git_commit"],
            "resume_from": manifest["resume_from"],
            "dataset": metadata["config"],
            "dataset_hashes": {split: item["content_sha256"] for split, item in metadata["splits"].items()},
        },
    )
    write_json(
        output_path / "wandb_run.json",
        {"id": run.id, "url": run.url, "project": config.project_name, "name": config.experiment_name},
    )
    print(f"W&B run: {run.url}", flush=True)
    return WandbLogger(run, references)
