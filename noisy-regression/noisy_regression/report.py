"""Produce loss and exact pass@k figures from distribution-only evaluation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noisy_regression.data import DatasetConfig
from noisy_regression.metrics import KS

REFERENCE_STYLES = {
    "uniform_256": ("Uniform 256", "#a1a1aa", ":"),
    "query_only_continuous_optimistic": ("Query-only (continuous)", "#64748b", "--"),
    "query_only_decoded_plugin_approximation": ("Query-only (quantized)", "#94a3b8", ":"),
    "bayesian_continuous_optimistic": ("Bayesian (continuous, optimistic)", "#168575", "--"),
    "ridge_decoded_gaussian_approximation": ("Ridge (decoded, approximate)", "#9e5bb5", ":"),
}


def render_report(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    references = json.loads((run / "references.json").read_text())
    metadata = json.loads((run / "dataset_metadata.json").read_text())
    dataset_config = DatasetConfig(**metadata["config"])
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    evaluations = [event for event in events if event["kind"] == "evaluation"]
    if not evaluations:
        raise ValueError("No evaluations have been recorded")
    final = evaluations[-1]
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
    steps = [event["step"] for event in evaluations]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for split, label in (("train", "Fixed training subset"), ("eval", "Held-out evaluation")):
        axes[0].plot(steps, [event[split]["answer_nll"]["mean"] for event in evaluations], marker=".", label=label)
    for name in (
        "query_only_continuous_optimistic",
        "bayesian_continuous_optimistic",
        "ridge_decoded_gaussian_approximation",
    ):
        label, color, style = REFERENCE_STYLES[name]
        axes[0].axhline(references[name]["answer_nll"]["mean"], linestyle=style, color=color, alpha=0.8, label=label)
    axes[0].set(xlabel="Optimizer steps", ylabel="NLL (nats / complete answer)")
    axes[0].legend(fontsize=7)
    for k in (1, 16, 256):
        axes[1].plot(
            steps,
            [event["eval"]["exact_pass"][str(k)]["mean"] for event in evaluations],
            label=f"Exact pass@{k} (all {metadata['config']['eval_count']})",
        )
    axes[1].set(xlabel="Optimizer steps", ylabel="Exact pass@k (log scale)", yscale="log", ylim=(1e-3, 1))
    axes[1].legend(fontsize=8)
    fig.savefig(run / "learning_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, target, source in zip(axes, ("Clean", "Noisy"), ("clean_exact_pass", "exact_pass"), strict=True):
        axis.plot(
            KS,
            [final["eval"][source][str(k)]["mean"] for k in KS],
            marker="o",
            label="Model",
        )
        for name in (
            "uniform_256",
            "query_only_continuous_optimistic",
            "bayesian_continuous_optimistic",
            "ridge_decoded_gaussian_approximation",
        ):
            label, color, style = REFERENCE_STYLES[name]
            axis.plot(
                KS,
                [references[name][source][str(k)]["mean"] for k in KS],
                linestyle=style,
                color=color,
                label=label,
            )
        axis.set(
            xscale="log",
            xlabel="k",
            ylabel="Exact pass@k",
            ylim=(0, 1),
            title=f"{target} target · {final['eval']['prompts']} prompts · step {final['step']:,}",
        )
        axis.set_xticks(KS, [str(k) for k in KS])
        axis.legend(fontsize=7)
    fig.savefig(run / "pass_at_k.png", dpi=180)
    plt.close(fig)
    rows = [
        "# Fixed-pool noisy regression SFT",
        "",
        f"Status: {'completed' if summary else 'in progress'}; last evaluated step {final['step']:,}. Trainable parameters: {manifest['parameter_count']:,}.",
        "",
        f"Shared context/query noise standard deviation: {dataset_config.sigma:g}. "
        "Reference predictors use these dataset noise settings.",
        "",
        f"Observed presentations: {final['presentations']:,}. "
        f"Recorded elapsed time: {(summary or final)['elapsed_seconds'] / 60:.2f} minutes. "
        f"Effective batch: {manifest['training']['batch_size']}; microbatch: {manifest['training']['micro_batch_size']}. "
        "Model/optimizer settings, exact decay groups, software versions and seeds: `manifest.json`.",
        "",
        "![Learning curves](learning_curves.png)",
        "",
        "![Pass at k](pass_at_k.png)",
        "",
        "| Predictor | Clean MSE | Noisy MSE | Clean NLL | Noisy NLL |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in [("Final model" if summary else "Latest model", final["eval"]), *references.items()]:
        label = REFERENCE_STYLES[name][0] if name in REFERENCE_STYLES else name
        errors = metric["predictive_mean_errors"]
        rows.append(
            f"| {label} | {errors['continuous_noiseless_signal']['mse']['mean']:.5f} | "
            f"{errors['continuous_noisy_outcome']['mse']['mean']:.5f} | "
            f"{metric['clean_answer_nll']['mean']:.5f} | {metric['answer_nll']['mean']:.5f} |"
        )
    if summary:
        rows += [
            "",
            f"Final checkpoint: `{summary['final_checkpoint']}`. Best held-out NLL: {summary['best']['answer_nll']:.5f} at step {summary['best']['step']}; checkpoint `{summary['best']['checkpoint']}`.",
        ]
    if "mismatched_context_control" in final["eval"]:
        mismatch = final["eval"]["mismatched_context_control"]["answer_nll"]["mean"]
        rows += [
            "",
            f"Mismatching context observations across tasks while retaining each query and target gives NLL {mismatch:.5f}, versus aligned-context NLL {final['eval']['answer_nll']['mean']:.5f}. This is a context-dependence diagnostic, not evidence of Bayes-optimal inference.",
        ]
    rows += [
        "",
        f"Evaluation enumerated the exact distribution over all 256 answers for each of "
        f"{final['eval']['prompts']} held-out prompts. No completions were sampled.",
        "",
        "MSE compares the exact probability-weighted grid mean with the continuous clean signal or noisy outcome. "
        "NLL and pass@k score the corresponding quantized target under that same distribution.",
        "",
        "| k | Clean exact pass@k | Noisy exact pass@k |",
        "|---|---:|---:|",
    ]
    for k in KS:
        rows.append(
            f"| {k} | {final['eval']['clean_exact_pass'][str(k)]['mean']:.5f} | "
            f"{final['eval']['exact_pass'][str(k)]['mean']:.5f} |"
        )
    rows += [
        "",
        "Exact pass@k averages 1-(1-p_target)^k across examples. No Monte Carlo estimator is needed. "
        "Repeated checkpoint comparisons share the same evaluation examples.",
        "",
        "Continuous-data references see more precise inputs/observations and are optimistic references. "
        "Decoded-data Gaussian/ridge predictors are plug-in approximations, not the exact posterior conditioned on quantized tokens. "
        "Endpoint bins integrate infinite tails. Predictive-mean errors against continuous noisy outcomes, "
        "decoded targets, and noiseless signals are separately recorded in the metrics.",
        "",
        "There is one held-out evaluation pool, reused for checkpoint selection, and no independent final test. "
        "This is one training seed and one frozen noisy pool. Low exact match alone does not establish model inadequacy; "
        "learning curves, context controls, reference gaps, entropy, and target stochasticity must be considered together.",
        "",
        "| Split / scalar family | Clipped / total | Fraction |",
        "|---|---:|---:|",
    ]
    for split, info in metadata["splits"].items():
        for name, clipping in info["clipping"].items():
            rows.append(
                f"| {split} / {name} | {clipping['below'] + clipping['above']} / {clipping['total_scalars']} | {clipping['fraction']:.8f} |"
            )
    rows += [
        "",
        "Dataset hashes, stable IDs and split-overlap audit: `dataset_metadata.json` and the source dataset's `metadata.json`/NPZ files. Per-prompt IDs and exact probabilities: `evaluation-*.npz`. Full checkpoint history and machine-readable metrics are retained.",
        "",
    ]
    (run / "report.md").write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    render_report(parser.parse_args().run)


if __name__ == "__main__":
    main()
