"""Produce loss/pass@k figures and a concise report from recorded results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from noisy_regression.data import load_pool, subset
from noisy_regression.metrics import KS
from noisy_regression.references import reference_report

REFERENCE_STYLES = {
    "uniform_256": ("Uniform 256", "#a1a1aa", ":"),
    "query_only_continuous_optimistic": ("Query-only (continuous)", "#64748b", "--"),
    "bayesian_continuous_optimistic": ("Bayesian (continuous, optimistic)", "#168575", "--"),
    "ridge_decoded_gaussian_approximation": ("Ridge (decoded, approximate)", "#9e5bb5", ":"),
}


def render_report(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    references = json.loads((run / "references.json").read_text())
    metadata = json.loads((run / "dataset_metadata.json").read_text())
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
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    generation = final["eval"]["generation"]
    sampled_mean_mse = generation["sampled_mean_mse"]
    generation_references = references
    if generation["prompts"] < metadata["config"]["eval_count"]:
        # Reference overlays use precisely the same tasks as sampled curves,
        # including when rendering intermediate 128-prompt evaluations.
        pools, _ = load_pool(manifest["data_path"])
        with np.load(run / f"evaluation-{final['step']:05d}.npz", allow_pickle=False) as archive:
            selected = subset(pools["eval"], archive["generation_indices"])
        generation_references = reference_report(selected)
    axis.plot(
        KS,
        [generation["exact_pass_same_subset"][str(k)]["mean"] for k in KS],
        marker="o",
        label="Model exact (same prompts)",
    )
    axis.errorbar(
        KS,
        [generation["generative_pass"][str(k)]["mean"] for k in KS],
        yerr=[1.96 * generation["generative_pass"][str(k)]["prompt_se"] for k in KS],
        marker=".",
        label="Generated estimate ±1.96 prompt SE",
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
            [generation_references[name]["exact_pass"][str(k)]["mean"] for k in KS],
            linestyle=style,
            color=color,
            label=label,
        )
    axis.set(
        xscale="log",
        xlabel="k",
        ylabel="pass@k",
        ylim=(0, 1),
        title=f"Same {generation['prompts']} held-out prompts, step {final['step']:,}",
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
        f"Observed presentations: {final['presentations']:,}. "
        f"Recorded elapsed time: {(summary or final)['elapsed_seconds'] / 60:.2f} minutes. "
        f"Effective batch: {manifest['training']['batch_size']}; microbatch: {manifest['training']['micro_batch_size']}. "
        "Model/optimizer settings, exact decay groups, software versions and seeds: `manifest.json`.",
        "",
        "![Learning curves](learning_curves.png)",
        "",
        "![Pass at k](pass_at_k.png)",
        "",
        "| Predictor | Answer NLL | Exact pass@1 | Exact pass@16 | Exact pass@256 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in [("Final model" if summary else "Latest model", final["eval"]), *references.items()]:
        rows.append(
            f"| {name} | {metric['answer_nll']['mean']:.5f} | {metric['exact_pass']['1']['mean']:.5f} | {metric['exact_pass']['16']['mean']:.5f} | {metric['exact_pass']['256']['mean']:.5f} |"
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
        f"Generation used {generation['prompts']} held-out prompts and 256 independent completions each. Invalid completions: {generation['invalid_completions']}; overlength: {generation['overlength_completions']}. Exact and generated estimates below refer to the same prompts.",
        "",
        f"The MSE of each prompt's mean of 256 decoded predictions against its continuous noiseless signal "
        f"(w·x_query) is {sampled_mean_mse['mean']:.6f} ± {sampled_mean_mse['prompt_se']:.6f} prompt SE "
        f"over {sampled_mean_mse['prompts']} prompts. Average predictions within each prompt before squaring the "
        "error, then average squared errors across prompts. This sampled mean includes Monte Carlo variability; "
        "the separately recorded exact-distribution predictive mean integrates over all 256 output bins.",
        "",
        "| k | Exact | Generated | Prompt SE | Conditional sampling SD of mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in KS:
        entry = generation["generative_pass"][str(k)]
        rows.append(
            f"| {k} | {generation['exact_pass_same_subset'][str(k)]['mean']:.5f} | {entry['mean']:.5f} | {entry['prompt_se']:.5f} | {entry['conditional_sampling_sd_of_mean']:.5f} |"
        )
    rows += [
        "",
        "The prompt SE treats regression examples as units. Normal intervals are approximate; "
        "conditional sampling SD integrates the estimator over Binomial(256,p_target) for each stored prompt. "
        "It does not include model-training seed variability. Repeated checkpoint comparisons share the same evaluation examples.",
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
        "Dataset hashes, stable IDs and split-overlap audit: `dataset_metadata.json` and the source dataset's `metadata.json`/NPZ files. Per-prompt probabilities and sampled completions: `evaluation-*.npz`. Full checkpoint history and machine-readable metrics are retained.",
        "",
    ]
    (run / "report.md").write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    render_report(parser.parse_args().run)


if __name__ == "__main__":
    main()
