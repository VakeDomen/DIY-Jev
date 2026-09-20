#!/usr/bin/env python3
"""Analyze saved benchmark runs: python3 benchmarks/analyze.py [result.json ...]."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from benchmark import RADAR_TASKS, RADAR_LABELS, PUBLISHED_OPENJEV, install_plot_dependencies

ROOT = Path(__file__).resolve().parent


def load_runs(paths):
    runs = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("predictions") or not data.get("tasks"):
            raise ValueError(f"{path}: expected a benchmark result with predictions and tasks")
        name = data.get("backend", {}).get("model")
        name = name or path.parent.name.removeprefix("results_")
        if any(run[0] == name for run in runs):
            name = f"{name} ({path.stem})"
        runs.append((name, data, path))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "charts")
    args = parser.parse_args()
    paths = args.results or sorted(ROOT.glob("results_*/*.json"))
    if not paths:
        parser.error("no results found in benchmarks/results_*; supply result JSON paths")
    runs = load_runs(paths)
    try:
        import matplotlib
    except ImportError:
        install_plot_dependencies()
        import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    names = [name for name, _, _ in runs]
    tasks = [task for task in RADAR_TASKS if any(task in d["tasks"] for _, d, _ in runs)]
    labels = [RADAR_LABELS[t].replace("\n", " ") for t in tasks]
    accuracy = np.array([[100 * d["tasks"][t]["accuracy"] if t in d["tasks"] else np.nan
                          for t in tasks] for _, d, _ in runs])
    baseline = np.array([100 * PUBLISHED_OPENJEV[t]["accuracy"] for t in tasks])
    hashes = [d.get("dataset", {}).get("selection_sha256") for _, d, _ in runs]
    matched = bool(hashes) and all(hashes) and len(set(hashes)) == 1
    comparability = "Identical recorded dataset selection" if matched else "Dataset selections differ or are unverified"
    generated = []

    def save(fig, filename):
        fig.savefig(out / filename, bbox_inches="tight")
        plt.close(fig)
        generated.append(filename)

    def heatmap(matrix, row_names, filename, title, diverging=False):
        fig, ax = plt.subplots(figsize=(13, max(3.5, len(row_names) * .65 + 2)))
        extent = max(1, float(np.nanmax(np.abs(matrix))))
        img = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto",
                        cmap="RdBu" if diverging else "YlGnBu",
                        vmin=-extent if diverging else 0, vmax=extent if diverging else 100)
        ax.set_xticks(range(len(tasks)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(row_names)), row_names)
        for i in range(len(row_names)):
            for j in range(len(tasks)):
                value = matrix[i, j]
                text = "—" if np.isnan(value) else (f"{value:+.1f}" if diverging else f"{value:.1f}%")
                color = "white" if np.isfinite(value) and (abs(value) > .65 * extent if diverging else value > 55) else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)
        ax.set_title(title + "\n" + comparability, pad=15)
        fig.colorbar(img, ax=ax, label="percentage points" if diverging else "accuracy (%)")
        save(fig, filename)

    heatmap(np.vstack([accuracy, baseline]), names + ["Published OpenJev NLI-4B"],
            "accuracy_by_task.png", "Top-1 accuracy by task")
    heatmap(accuracy - baseline, names, "delta_vs_openjev.png",
            "Accuracy difference from published OpenJev (positive is better)", True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, data, _ in runs:
        predictions = data["predictions"]
        bins = [[] for _ in range(10)]
        for p in predictions:
            confidence = float(p["probabilities"][p["predicted"]])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"{name}: invalid predicted-class probability")
            bins[min(9, int(confidence * 10))].append((confidence, p["gold"] == p["predicted"]))
        populated = [b for b in bins if b]
        x = [sum(c for c, _ in b) / len(b) for b in populated]
        y = [sum(ok for _, ok in b) / len(b) for b in populated]
        axes[0].plot(x, y, marker="o", label=name)
        axes[1].stairs([len(b) / len(predictions) for b in bins], np.linspace(0, 1, 11), label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.4, label="Perfect calibration")
    axes[0].set(xlabel="Mean predicted-class probability", ylabel="Observed accuracy",
                title="Reliability (10 equal-width bins)", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Predicted-class probability", ylabel="Fraction of predictions",
                title="Confidence distribution", xlim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Calibration across the observed task mix; small bins can be noisy")
    fig.tight_layout()
    save(fig, "calibration.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, data, _ in runs:
        latencies = np.sort([p["latency_ms"] for p in data["predictions"]])
        axes[0].plot(latencies, np.arange(1, len(latencies) + 1) / len(latencies), label=name)
        summary = data["summary"]
        macro = np.mean([data["tasks"][t]["accuracy"] for t in tasks if t in data["tasks"]]) * 100
        rate = summary["requests_per_second"]
        axes[1].scatter(rate, macro, s=65)
        axes[1].annotate(name, (rate, macro), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="HTTP latency including queueing (ms)", ylabel="Fraction completed",
                title="Latency CDF (first repetition)", xscale="log", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Measured requests / second", ylabel="Macro task accuracy (%)",
                title="Observed accuracy / throughput", ylim=(0, 100))
    fig.suptitle("Hardware, concurrency and workload affect speed; this is not a controlled speed ranking")
    fig.tight_layout()
    save(fig, "latency_and_throughput.png")

    # Machine-readable task table, including observed sample sizes and chance accuracy.
    with (out / "task_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "task", "n", "accuracy", "chance_accuracy", "openjev_accuracy",
                         "delta", "published_n", "nll", "brier", "ece_10", "latency_p50_ms"])
        for name, data, _ in runs:
            for task in tasks:
                if task not in data["tasks"]:
                    continue
                s = data["tasks"][task]
                ps = [p for p in data["predictions"] if p["task"] == task]
                chance = sum(1 / len(p["probabilities"]) for p in ps) / len(ps)
                writer.writerow([name, task, len(ps), s["accuracy"], chance,
                                 PUBLISHED_OPENJEV[task]["accuracy"],
                                 s["accuracy"] - PUBLISHED_OPENJEV[task]["accuracy"],
                                 PUBLISHED_OPENJEV[task]["examples"], s["nll"], s["brier"],
                                 s["ece_10"], s["latency_ms"]["p50"]])

    lines = ["# Benchmark analysis", "", comparability + ".", "",
             "Macro accuracy weights each available task equally; micro accuracy weights every question equally.",
             "Published OpenJev scores are full-suite reference values, not paired predictions.",
             "GSM8K distractor ordering, dataset revisions and subset selection can affect comparability.", "",
             "| Model | Questions | Tasks | Micro accuracy | Macro accuracy | NLL | ECE | Requests/s | Concurrency |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, data, path in runs:
        s = data["summary"]
        present = [t for t in tasks if t in data["tasks"]]
        macro = sum(data["tasks"][t]["accuracy"] for t in present) / len(present)
        lines.append(f"| {name} | {len(data['predictions'])} | {len(present)} | {s['accuracy']:.2%} | "
                     f"{macro:.2%} | {s['nll']:.3f} | {s['ece_10']:.3f} | "
                     f"{s['requests_per_second']:.2f} | {data['settings']['concurrency']} |")
    lines += ["", "## Charts", ""] + [f"![{f}]({f})" for f in generated]
    signatures = {}
    findings = []
    for name, data, _ in runs:
        # Ignore timing and row order when detecting duplicated model outputs.
        records = sorted((p["task"], p["id"], p["gold"], p["predicted"], p["probabilities"])
                         for p in data["predictions"])
        signature = hashlib.sha256(json.dumps(records).encode()).hexdigest()
        if signature in signatures:
            findings.append(f"- {name} and {signatures[signature]} have identical predictions and "
                            "probability vectors for every example. Verify the served model identity "
                            "before treating these as independent model measurements.")
        else:
            signatures[signature] = name
    if findings:
        lines += ["", "## Data checks", ""] + findings
        for finding in findings:
            print(finding)
    lines += ["", "## Sources", ""] + [f"- {path}" for _, _, path in runs]
    lines += ["- https://huggingface.co/AlexWortega/openjev/blob/main/results/qwen4b_all.json",
              "- https://huggingface.co/AlexWortega/openjev/blob/main/results/qwen4b_extra_mc.json"]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed {len(runs)} runs; wrote {len(generated)} charts, task_metrics.csv and README.md to {out}")


if __name__ == "__main__":
    main()
