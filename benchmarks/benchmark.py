#!/usr/bin/env python3
"""Benchmark a Granite Jev server against OpenJev's published MC results.

The input is JSONL. Each row must contain:

    {"id": "optional", "q": "question", "opts": ["a", "b"], "gold": 0}

`question`/`options`/`label` are accepted as aliases. `label` may be an option
index or the exact option text. Optional `task` and `hypotheses` fields retain
the radar-suite grouping and provenance of OpenJev's task construction. The
server backend needs only Python's standard library. Other commands document
their optional dependencies when an import is missing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_INSTRUCTION = "Which option correctly answers the question?"
BENCHMARK_DATA_DIR = Path(__file__).resolve().parent / "data"
LOCAL_DEPS_DIR = Path(__file__).resolve().parent / ".deps"
DEFAULT_DATA_PATH = BENCHMARK_DATA_DIR / "radar-full.jsonl"
PUBLISHED_RESULT_SOURCES = (
    "https://huggingface.co/AlexWortega/openjev/blob/main/results/qwen4b_all.json",
    "https://huggingface.co/AlexWortega/openjev/blob/main/results/qwen4b_extra_mc.json",
)
RADAR_TASKS = (
    "mmlu",
    "gpqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "hellaswag",
    "gsm8k_mc4",
    "gsm8k_mc10",
    "chess",
)
RADAR_LABELS = {
    "mmlu": "MMLU",
    "gpqa": "GPQA\nDiamond",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "winogrande": "WinoGrande",
    "hellaswag": "HellaSwag",
    "gsm8k_mc4": "GSM8K\n4 choices",
    "gsm8k_mc10": "GSM8K\n10 choices",
    "chess": "Chess\n4 legal moves",
}
PUBLISHED_OPENJEV = {
    "mmlu": {"examples": 14042, "accuracy": 0.47151402934054976},
    "gpqa": {"examples": 198, "accuracy": 0.2727272727272727},
    "arc_easy": {"examples": 2376, "accuracy": 0.7685185185185185},
    "arc_challenge": {"examples": 1172, "accuracy": 0.5921501706484642},
    "winogrande": {"examples": 1267, "accuracy": 0.585635359116022},
    "hellaswag": {"examples": 10042, "accuracy": 0.31686914957179846},
    "gsm8k_mc4": {"examples": 1319, "accuracy": 0.39196360879454134},
    "gsm8k_mc10": {"examples": 1319, "accuracy": 0.17513267626990145},
    "chess": {"examples": 500, "accuracy": 0.24},
}

if LOCAL_DEPS_DIR.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS_DIR))


@dataclass(frozen=True)
class Example:
    id: str
    task: str
    question: str
    options: tuple[str, ...]
    gold: int
    hypotheses: tuple[str, ...] | None = None


@dataclass
class Prediction:
    id: str
    task: str
    gold: int
    predicted: int
    probabilities: list[float]
    latency_ms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "gold": self.gold,
            "predicted": self.predicted,
            "probabilities": self.probabilities,
            "latency_ms": self.latency_ms,
            "correct": self.gold == self.predicted,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    server = sub.add_parser("server", help="benchmark a running Granite Jev server")
    add_run_arguments(server)
    server.add_argument("--url", default="http://127.0.0.1:8080/v1/evaluate")
    server.add_argument("--timeout", type=float, default=120.0)
    server.add_argument("--concurrency", type=positive_int, default=1)
    server.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    server.add_argument(
        "--state-format",
        choices=("question", "structured"),
        default="question",
        help="send state as a string or as {question: ...}",
    )

    compare = sub.add_parser(
        "compare", help="compare a saved server result with published OpenJev numbers"
    )
    compare.add_argument("result", type=Path)

    prepare = sub.add_parser(
        "prepare", help="export one task or the chart's complete nine-axis radar suite"
    )
    prepare.add_argument(
        "task",
        choices=(*RADAR_TASKS, "radar"),
    )
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--limit", type=positive_int)
    prepare.add_argument("--seed", type=int, default=0)

    radar = sub.add_parser("radar", help="plot per-task accuracy from result files")
    radar.add_argument("results", nargs="+", type=Path)
    radar.add_argument("--output", type=Path, default=Path("benchmark-results/radar.png"))
    radar.add_argument("--title", default="Granite Jev vs. OpenJev")
    radar.add_argument(
        "--label",
        action="append",
        help="series label; repeat once per result file (defaults to backend names)",
    )
    return parser


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"JSONL fixture (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=nonnegative_int, default=3)
    parser.add_argument("--repeats", type=positive_int, default=1)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def load_examples(path: Path, limit: int | None, seed: int) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                question = row.get("q", row.get("question"))
                options = row.get("opts", row.get("options"))
                label = row.get("gold", row.get("label"))
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("q/question must be a non-empty string")
                if not isinstance(options, list) or len(options) < 2:
                    raise ValueError("opts/options must contain at least two entries")
                if not all(isinstance(option, str) for option in options):
                    raise ValueError("every option must be a string")
                hypotheses = row.get("hypotheses")
                if hypotheses is not None:
                    if (
                        not isinstance(hypotheses, list)
                        or len(hypotheses) != len(options)
                        or not all(isinstance(value, str) for value in hypotheses)
                    ):
                        raise ValueError(
                            "hypotheses must be a string list matching the options"
                        )
                if isinstance(label, str):
                    label = options.index(label)
                if isinstance(label, bool) or not isinstance(label, int):
                    raise ValueError("gold/label must be an integer or exact option text")
                if not 0 <= label < len(options):
                    raise ValueError("gold/label is outside the option list")
                examples.append(
                    Example(
                        id=str(row.get("id", line_number - 1)),
                        task=str(row.get("task", "all")),
                        question=question.strip(),
                        options=tuple(options),
                        gold=label,
                        hypotheses=tuple(hypotheses) if hypotheses is not None else None,
                    )
                )
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

    if not examples:
        raise ValueError(f"{path} contains no examples")
    if limit is not None and limit < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:limit]
    return examples


class ServerBackend:
    def __init__(
        self,
        url: str,
        timeout: float,
        instruction: str,
        state_format: str,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.instruction = instruction
        self.state_format = state_format

    def predict(self, example: Example) -> Prediction:
        keys = [f"option_{index}" for index in range(len(example.options))]
        state: Any = example.question
        if self.state_format == "structured":
            state = {"question": example.question}
        body = {
            "state": state,
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": self.instruction,
                    "criteria": dict(zip(keys, example.options)),
                }
            },
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {error.code} from {self.url}: {detail}") from error
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            result = payload.get("result", payload)
            answer = result["answers"]["answer"]
            probabilities = [float(answer["probabilities"][key]) for key in keys]
            predicted = keys.index(answer["choice"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid server response: {payload!r}") from error
        if any(value < 0 or not math.isfinite(value) for value in probabilities):
            raise RuntimeError(f"invalid server probabilities: {probabilities!r}")
        total = sum(probabilities)
        if total <= 0:
            raise RuntimeError(f"server probabilities have zero mass: {probabilities!r}")
        probabilities = [value / total for value in probabilities]
        return Prediction(
            example.id,
            example.task,
            example.gold,
            predicted,
            probabilities,
            latency_ms,
        )


def run_backend(
    examples: Sequence[Example],
    predict: Callable[[Example], Prediction],
    backend: dict[str, Any],
    warmup: int,
    repeats: int,
    concurrency: int,
) -> dict[str, Any]:
    for index in range(warmup):
        predict(examples[index % len(examples)])

    started = time.perf_counter()
    jobs = [
        (repeat, index, example)
        for repeat in range(repeats)
        for index, example in enumerate(examples)
    ]
    if concurrency == 1:
        completed = [(repeat, index, predict(example)) for repeat, index, example in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(predict, example): (repeat, index)
                for repeat, index, example in jobs
            }
            completed = []
            for future in concurrent.futures.as_completed(futures):
                repeat, index = futures[future]
                completed.append((repeat, index, future.result()))
    wall_seconds = time.perf_counter() - started
    completed.sort(key=lambda item: (item[0], item[1]))

    primary = [prediction for repeat, _, prediction in completed if repeat == 0]
    all_latencies = [prediction.latency_ms for _, _, prediction in completed]
    tasks: dict[str, Any] = {}
    for task in dict.fromkeys(example.task for example in examples):
        task_predictions = [prediction for prediction in primary if prediction.task == task]
        task_latencies = [
            prediction.latency_ms
            for _, _, prediction in completed
            if prediction.task == task
        ]
        tasks[task] = summarize(task_predictions, task_latencies)
    return {
        "format_version": 2,
        "backend": backend,
        "settings": {
            "examples": len(examples),
            "warmup": warmup,
            "repeats": repeats,
            "concurrency": concurrency,
        },
        "summary": summarize(primary, all_latencies, wall_seconds, len(completed)),
        "tasks": tasks,
        "predictions": [prediction.to_json() for prediction in primary],
    }


def summarize(
    predictions: Sequence[Prediction],
    latencies: Sequence[float],
    wall_seconds: float | None = None,
    requests: int | None = None,
) -> dict[str, Any]:
    correct = [prediction.gold == prediction.predicted for prediction in predictions]
    top3 = [
        prediction.gold
        in sorted(
            range(len(prediction.probabilities)),
            key=prediction.probabilities.__getitem__,
            reverse=True,
        )[:3]
        for prediction in predictions
    ]
    nll = [
        -math.log(max(prediction.probabilities[prediction.gold], 1e-15))
        for prediction in predictions
    ]
    brier = []
    confidences = []
    for prediction in predictions:
        confidences.append(max(prediction.probabilities))
        brier.append(
            sum(
                (probability - float(index == prediction.gold)) ** 2
                for index, probability in enumerate(prediction.probabilities)
            )
        )
    summary = {
        "examples": len(predictions),
        "accuracy": statistics.fmean(correct),
        "top3_accuracy": statistics.fmean(top3),
        "nll": statistics.fmean(nll),
        "brier": statistics.fmean(brier),
        "ece_10": expected_calibration_error(confidences, correct, 10),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "min": min(latencies),
            "max": max(latencies),
        },
    }
    if wall_seconds is not None and requests is not None:
        summary["wall_seconds"] = wall_seconds
        summary["requests_per_second"] = requests / wall_seconds
    return summary


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int
) -> float:
    total = len(confidences)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bin_index == bins - 1 and confidence == 1.0)
        ]
        if members:
            accuracy = statistics.fmean(correct[index] for index in members)
            confidence = statistics.fmean(confidences[index] for index in members)
            error += len(members) / total * abs(accuracy - confidence)
    return error


def percentile(values: Sequence[float], percentage: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def save_and_print(result: dict[str, Any], output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = result["summary"]
    latency = summary["latency_ms"]
    print(f"backend:    {result['backend']['name']}")
    print(f"examples:   {result['settings']['examples']}")
    print(f"accuracy:   {summary['accuracy']:.4f}")
    print(f"top-3:      {summary['top3_accuracy']:.4f}")
    print(f"NLL:        {summary['nll']:.4f}")
    print(f"Brier:      {summary['brier']:.4f}")
    print(f"ECE (10):   {summary['ece_10']:.4f}")
    print(
        f"latency ms: p50={latency['p50']:.2f} p95={latency['p95']:.2f} "
        f"p99={latency['p99']:.2f} mean={latency['mean']:.2f}"
    )
    print(f"throughput: {summary['requests_per_second']:.2f} requests/s")
    if any(task in result.get("tasks", {}) for task in RADAR_TASKS):
        print_published_comparison(result)
    if output:
        print(f"saved:      {output}")


def published_comparison(result: dict[str, Any]) -> dict[str, Any]:
    comparison = {}
    for task in RADAR_TASKS:
        if task not in result.get("tasks", {}):
            continue
        measured = result["tasks"][task]
        published = PUBLISHED_OPENJEV[task]
        comparison[task] = {
            "server_examples": measured["examples"],
            "published_examples": published["examples"],
            "same_sample_size": measured["examples"] == published["examples"],
            "server_accuracy": measured["accuracy"],
            "published_openjev_accuracy": published["accuracy"],
            "delta": measured["accuracy"] - published["accuracy"],
        }
    return comparison


def print_published_comparison(result: dict[str, Any]) -> None:
    comparison = result.get("published_comparison") or published_comparison(result)
    if not comparison:
        print("\nno radar tasks available for published comparison")
        return
    print("\nserver vs. published OpenJev NLI-4B (top-1 accuracy):")
    print(f"{'task':22s} {'server':>8s} {'OpenJev':>8s} {'delta':>9s} {'n':>13s}")
    for task in RADAR_TASKS:
        if task not in comparison:
            continue
        values = comparison[task]
        sample = f"{values['server_examples']}/{values['published_examples']}"
        marker = "" if values["same_sample_size"] else " *"
        print(
            f"{RADAR_LABELS[task].replace(chr(10), ' '):22s} "
            f"{values['server_accuracy']:8.4f} "
            f"{values['published_openjev_accuracy']:8.4f} "
            f"{values['delta']:+9.4f} {sample:>11s}{marker}"
        )
    if any(not values["same_sample_size"] for values in comparison.values()):
        print(
            "* subset size differs from the published full-suite result; "
            "delta is indicative only"
        )


def compare_result(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    print(f"server:     {result['backend']['name']} ({path})")
    print_published_comparison(result)


def plot_radar(
    result_paths: Sequence[Path], output: Path, title: str, labels: Sequence[str] | None
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("radar plotting needs matplotlib and numpy") from error

    if labels is not None and len(labels) != len(result_paths):
        raise ValueError("--label must be omitted or repeated once per result file")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    missing = [
        f"{path}: {task}"
        for path, result in zip(result_paths, results)
        for task in RADAR_TASKS
        if task not in result.get("tasks", {})
    ]
    if missing:
        raise ValueError("missing radar task summaries: " + ", ".join(missing))

    series_labels = list(labels) if labels else [result["backend"]["name"] for result in results]
    angles = np.linspace(0, 2 * np.pi, len(RADAR_TASKS), endpoint=False)
    closed_angles = np.r_[angles, angles[:1]]
    figure = plt.figure(figsize=(9.2, 10.4), facecolor="white")
    axis = figure.add_subplot(111, polar=True)
    figure.subplots_adjust(top=0.86, bottom=0.17)
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_facecolor("#f4f5f7")
    axis.set_ylim(0, 100)
    axis.set_yticks([20, 40, 60, 80, 100])
    axis.set_yticklabels([f"{value}%" for value in [20, 40, 60, 80, 100]], color="#9aa0a6")
    axis.set_xticks(angles)
    axis.set_xticklabels([RADAR_LABELS[task] for task in RADAR_TASKS])
    axis.grid(color="#d0d4d9", linewidth=0.8)
    axis.spines["polar"].set_color("#c8ccd1")

    published_values = np.array(
        [100 * PUBLISHED_OPENJEV[task]["accuracy"] for task in RADAR_TASKS],
        dtype=float,
    )
    closed_published = np.r_[published_values, published_values[:1]]
    axis.plot(
        closed_angles,
        closed_published,
        color="#4a4a4a",
        linewidth=2,
        marker="o",
        markersize=4,
        label="OpenJev NLI-4B · published",
    )
    axis.fill(closed_angles, closed_published, color="#4a4a4a", alpha=0.06)

    for result, label in zip(results, series_labels):
        values = np.array(
            [100 * result["tasks"][task]["accuracy"] for task in RADAR_TASKS],
            dtype=float,
        )
        closed_values = np.r_[values, values[:1]]
        axis.plot(closed_angles, closed_values, linewidth=2, marker="o", markersize=4, label=label)
        axis.fill(closed_angles, closed_values, alpha=0.06)
    axis.set_title(title, fontsize=17, fontweight="bold", pad=42)
    figure.text(
        0.5,
        0.905,
        "multiple-choice top-1 accuracy, common 0–100% scale",
        ha="center",
        fontsize=9.5,
        color="#666",
    )
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False)
    figure.text(
        0.5,
        0.035,
        "OpenJev values are copied from its published result files. "
        "Polygon area is not an aggregate score.",
        ha="center",
        fontsize=8,
        color="#777",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)
    print(f"wrote {output}")


def install_data_dependencies() -> None:
    print(f"installing benchmark data dependencies into {LOCAL_DEPS_DIR}")
    LOCAL_DEPS_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--upgrade",
        "--target",
        str(LOCAL_DEPS_DIR),
        "datasets",
        "python-chess",
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "could not install benchmark data dependencies automatically; "
            f"run {' '.join(command)}"
        ) from error
    if str(LOCAL_DEPS_DIR) not in sys.path:
        sys.path.insert(0, str(LOCAL_DEPS_DIR))
    importlib.invalidate_caches()


def prepare_dataset(task: str, output: Path, limit: int | None, seed: int) -> None:
    try:
        from datasets import load_dataset as huggingface_load_dataset
        import chess  # noqa: F401 -- validate both data dependencies up front
    except ImportError:
        install_data_dependencies()
        try:
            from datasets import load_dataset as huggingface_load_dataset
            import chess  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "benchmark data dependencies remain unavailable after installation"
            ) from error

    def load_dataset(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("cache_dir", str(BENCHMARK_DATA_DIR / "huggingface"))
        return huggingface_load_dataset(*args, **kwargs)

    tasks = RADAR_TASKS if task == "radar" else (task,)
    rows: list[dict[str, Any]] = []
    for task_name in tasks:
        task_rows = build_task_rows(task_name, load_dataset, limit, seed)
        rows.extend(task_rows)
        print(f"prepared {len(task_rows):5d} {RADAR_LABELS[task_name].replace(chr(10), ' ')}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {output}")


def build_task_rows(
    task: str,
    load_dataset: Callable[..., Any],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if task == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split="test")
        indices = selected_indices(len(dataset), limit, seed)
        return [
            benchmark_row(
                task,
                index,
                dataset[index]["question"].strip(),
                [choice.strip() for choice in dataset[index]["choices"]],
                int(dataset[index]["answer"]),
            )
            for index in indices
        ]
    if task == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        indices = selected_indices(len(dataset), limit, seed)
        rows = []
        for index in indices:
            item = dataset[index]
            options = [
                item["Correct Answer"],
                item["Incorrect Answer 1"],
                item["Incorrect Answer 2"],
                item["Incorrect Answer 3"],
            ]
            rows.append(
                benchmark_row(
                    task,
                    index,
                    item["Question"].strip(),
                    [option.strip() for option in options],
                    0,
                )
            )
        return rows
    elif task in ("arc_easy", "arc_challenge"):
        config = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        dataset = load_dataset("allenai/ai2_arc", config, split="test")
        valid = []
        for index in range(len(dataset)):
            item = dataset[index]
            labels = item["choices"]["label"]
            if item["answerKey"] in labels:
                valid.append(index)
        selected = selected_values(valid, limit, seed)
        return [
            benchmark_row(
                task,
                index,
                dataset[index]["question"].strip(),
                [choice.strip() for choice in dataset[index]["choices"]["text"]],
                dataset[index]["choices"]["label"].index(dataset[index]["answerKey"]),
            )
            for index in selected
        ]
    elif task == "winogrande":
        dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        rows = []
        for index in selected_indices(len(dataset), limit, seed):
            item = dataset[index]
            options = [item["option1"], item["option2"]]
            rows.append(
                benchmark_row(
                    task,
                    index,
                    item["sentence"],
                    options,
                    int(item["answer"]) - 1,
                    [item["sentence"].replace("_", option) for option in options],
                )
            )
        return rows
    elif task == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        rows = []
        for index in selected_indices(len(dataset), limit, seed):
            item = dataset[index]
            context = (
                (item["ctx_a"] + " " + item["ctx_b"].capitalize()).strip()
                if item["ctx_b"]
                else item["ctx_a"].strip()
            )
            options = [ending.strip() for ending in item["endings"]]
            rows.append(
                benchmark_row(
                    task,
                    index,
                    f"{item['activity_label']}: {context}",
                    options,
                    int(item["label"]),
                    options,
                )
            )
        return rows
    elif task in ("gsm8k_mc4", "gsm8k_mc10"):
        choice_count = 4 if task == "gsm8k_mc4" else 10
        return build_gsm8k_rows(task, choice_count, load_dataset, limit, seed)
    elif task == "chess":
        return build_chess_rows(min(limit, 500) if limit is not None else 500, seed)
    raise ValueError(f"unsupported task: {task}")


def benchmark_row(
    task: str,
    index: int,
    question: str,
    options: Sequence[str],
    gold: int,
    hypotheses: Sequence[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": f"{task}:{index}",
        "task": task,
        "q": question,
        "opts": list(options),
        "gold": gold,
    }
    if hypotheses is not None:
        row["hypotheses"] = list(hypotheses)
    return row


def selected_indices(size: int, limit: int | None, seed: int) -> list[int]:
    return selected_values(list(range(size)), limit, seed)


def selected_values(values: list[int], limit: int | None, seed: int) -> list[int]:
    if limit is not None and limit < len(values):
        random.Random(seed).shuffle(values)
        return values[:limit]
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def examples_sha256(examples: Sequence[Example]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        row = {
            "id": example.id,
            "task": example.task,
            "question": example.question,
            "options": example.options,
            "gold": example.gold,
            "hypotheses": example.hypotheses,
        }
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text: str) -> str:
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    value = match.group(1) if match else (NUMBER_RE.findall(text) or [""])[-1]
    value = value.replace(",", "").rstrip(".")
    if not value:
        raise ValueError(f"could not extract GSM8K answer from {text!r}")
    number = float(value)
    return str(int(number)) if number == int(number) else str(number)


def build_gsm8k_rows(
    task: str,
    choice_count: int,
    load_dataset: Callable[..., Any],
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for index in selected_indices(len(dataset), limit, seed):
        item = dataset[index]
        gold = extract_number(item["answer"])
        gold_value = float(gold)
        candidates: set[str] = set()
        generators = [
            lambda: gold_value + rng.choice([1, 2, 3, 5, 10]),
            lambda: gold_value - rng.choice([1, 2, 3, 5, 10]),
            lambda: gold_value * 2,
            lambda: gold_value / 2,
            lambda: gold_value + rng.choice([4, 6, 7, 8, 9, 12, 15, 20, 25, 50]),
            lambda: gold_value * 10,
            lambda: gold_value * rng.choice([3, 4, 5]),
            lambda: gold_value - rng.choice([4, 6, 7, 8, 9, 12, 15, 20, 25, 50]),
            lambda: abs(gold_value) + rng.randint(100, 999),
        ]
        generator_index = 0
        while len(candidates) < choice_count - 1 and generator_index < 200:
            value = generators[generator_index % len(generators)]()
            generator_index += 1
            rendered = str(int(value)) if float(value) == int(value) else f"{value:.2f}"
            if rendered != gold and rendered not in candidates and value >= 0:
                candidates.add(rendered)
        if len(candidates) != choice_count - 1:
            raise RuntimeError(
                f"could not generate {choice_count - 1} distractors for {task}:{index}"
            )
        # Sort before assigning seeded shuffle keys so PYTHONHASHSEED cannot
        # change the generated fixture through set iteration order.
        options = [gold] + sorted(sorted(candidates), key=lambda _: rng.random())
        order = list(range(len(options)))
        rng.shuffle(order)
        options = [options[position] for position in order]
        rows.append(
            benchmark_row(
                task,
                index,
                item["question"].strip(),
                options,
                options.index(gold),
                [f"The answer is {option}." for option in options],
            )
        )
    return rows


def build_chess_rows(count: int, seed: int) -> list[dict[str, Any]]:
    try:
        import chess
    except ImportError as error:
        raise RuntimeError("preparing chess needs python-chess") from error

    rng = random.Random(seed)
    rows = []
    while len(rows) < count:
        board = chess.Board()
        for _ in range(rng.randint(6, 40)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        legal = list(board.legal_moves)
        if len(legal) < 2 or board.is_game_over():
            continue
        legal_san = {board.san(move) for move in legal}
        good = board.san(rng.choice(legal))
        bad: set[str] = set()
        tries = 0
        while len(bad) < 3 and tries < 500:
            tries += 1
            square = rng.choice(
                [
                    value
                    for value in chess.SQUARES
                    if board.piece_at(value)
                    and board.piece_at(value).color == board.turn
                ]
            )
            piece = board.piece_at(square)
            destination = rng.choice(chess.SQUARES)
            if destination == square or (
                board.piece_at(destination)
                and board.piece_at(destination).color == board.turn
            ):
                continue
            capture = board.piece_at(destination) is not None
            if piece.piece_type == chess.PAWN:
                san = (
                    (chess.square_name(square)[0] + "x" if capture else "")
                    + chess.square_name(destination)
                )
            else:
                san = (
                    chess.piece_symbol(piece.piece_type).upper()
                    + ("x" if capture else "")
                    + chess.square_name(destination)
                )
            if san not in legal_san and san != good:
                bad.add(san)
        if len(bad) < 3:
            continue
        options = [good] + sorted(bad)
        rng.shuffle(options)
        history = chess.Board().variation_san(board.move_stack)
        question = (
            f"Chess position after the moves: {history}\nFEN: {board.fen()}\n"
            f"{'White' if board.turn else 'Black'} to move. Which of the following "
            "moves is legal in this position?"
        )
        rows.append(
            benchmark_row(
                task="chess",
                index=len(rows),
                question=question,
                options=options,
                gold=options.index(good),
            )
        )
    return rows


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "compare":
            compare_result(args.result)
            return 0
        if args.command == "prepare":
            prepare_dataset(args.task, args.output, args.limit, args.seed)
            return 0
        if args.command == "radar":
            plot_radar(args.results, args.output, args.title, args.label)
            return 0

        data_path = args.data.resolve()
        if not data_path.exists():
            if data_path != DEFAULT_DATA_PATH:
                raise ValueError(f"dataset fixture does not exist: {data_path}")
            print(f"benchmark fixture not found; downloading to {data_path}")
            prepare_dataset("radar", data_path, None, args.seed)
        args.data = data_path

        examples = load_examples(args.data, args.limit, args.seed)
        backend = ServerBackend(args.url, args.timeout, args.instruction, args.state_format)
        result = run_backend(
            examples,
            backend.predict,
            {
                "name": "granite-jev-server",
                "url": args.url,
                "instruction": args.instruction,
                "state_format": args.state_format,
            },
            args.warmup,
            args.repeats,
            args.concurrency,
        )
        result["dataset"] = {
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "selection_sha256": examples_sha256(examples),
        }
        result["published_comparison"] = published_comparison(result)
        result["published_reference"] = {
            "name": "AlexWortega/openjev qwen3.5-4b-nli",
            "metric": "rerank_acc",
            "sources": list(PUBLISHED_RESULT_SOURCES),
        }
        save_and_print(result, args.output)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
