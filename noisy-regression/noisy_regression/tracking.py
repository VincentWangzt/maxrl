"""Curated W&B curves; full evaluation diagnostics stay in JSON/NPZ artifacts."""

from dataclasses import dataclass

from noisy_regression.data import write_json

DASHBOARD_KS = (1, 16, 256)
# One no-context baseline, one approximation with the model's input precision,
# and one optimistic continuous-data benchmark. All five remain in references.json.
DASHBOARD_REFERENCES = {
    "query_only": "query_only_decoded_plugin_approximation",
    "ridge_quantized": "ridge_decoded_gaussian_approximation",
    "bayes_continuous": "bayesian_continuous_optimistic",
}


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    project_name: str
    experiment_name: str


def reference_metrics(references):
    """Six baseline curves grouped beside the model metrics they explain."""
    metrics = {}
    for label, name in DASHBOARD_REFERENCES.items():
        reference = references[name]
        metrics[f"likelihood/{label}_answer_nll"] = reference["answer_nll"]["mean"]
        metrics[f"regression/{label}_signal_mse"] = reference["predictive_mean_errors"]["continuous_noiseless_signal"][
            "mse"
        ]["mean"]
    return metrics


def event_metrics(event):
    metrics = {
        "progress/optimizer_step": event["step"],
        "progress/training_examples_seen": event["presentations"],
        "progress/elapsed_seconds": event["elapsed_seconds"],
    }
    if event["kind"] == "optimization":
        metrics.update(
            {
                "train/batch_answer_nll": event["answer_nll"],
                "train/gradient_norm_before_clip": event["gradient_norm_before_clip"],
                "train/learning_rate": event["learning_rate"],
            }
        )
    elif event["kind"] == "evaluation":
        evaluation = event["eval"]
        generation = evaluation["generation"]
        metrics.update(
            {
                "likelihood/eval_answer_nll": evaluation["answer_nll"]["mean"],
                "likelihood/train_answer_nll": event["train"]["answer_nll"]["mean"],
                "regression/model_signal_mse": evaluation["predictive_mean_errors"]["continuous_noiseless_signal"][
                    "mse"
                ]["mean"],
                "regression/sampled_signal_mse_256": generation["sampled_mean_mse"]["mean"],
                "diagnostics/predictive_entropy_nats": evaluation["entropy_nats_per_answer"]["mean"],
                # This is the only changing evaluation-size counter: the final
                # generation event expands from the fixed subset to the full pool.
                "progress/generation_prompts": generation["prompts"],
            }
        )
        for k in DASHBOARD_KS:
            metrics[f"pass_exact/pass@{k}"] = evaluation["exact_pass"][str(k)]["mean"]
            metrics[f"pass_sampled/pass@{k}"] = generation["generative_pass"][str(k)]["mean"]
        if "mismatched_context_control" in evaluation:
            metrics["diagnostics/context_shuffle_nll_increase"] = (
                evaluation["mismatched_context_control"]["answer_nll"]["mean"] - evaluation["answer_nll"]["mean"]
            )
    else:
        raise ValueError(f"Unknown metric event kind: {event['kind']}")
    return metrics


class WandbLogger:
    def __init__(self, run, references):
        self.run = run
        self.references = reference_metrics(references)
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
        self.run.summary.update(
            {
                "result/best_answer_nll": summary["best"]["answer_nll"],
                "result/best_step": summary["best"]["step"],
                "result/best_checkpoint": summary["best"]["checkpoint"],
                "result/final_checkpoint": summary["final_checkpoint"],
            }
        )
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
            "dashboard_schema_version": 2,
            "dashboard_pass_k": list(DASHBOARD_KS),
            "dashboard_references": DASHBOARD_REFERENCES,
            "train_evaluation_prompts": len(manifest["train_evaluation_ids"]),
            "periodic_generation_prompts": len(manifest["periodic_generation_ids"]),
            "final_generation_prompts": metadata["config"]["eval_count"],
        },
    )
    write_json(
        output_path / "wandb_run.json",
        {"id": run.id, "url": run.url, "project": config.project_name, "name": config.experiment_name},
    )
    print(f"W&B run: {run.url}", flush=True)
    return WandbLogger(run, references)
