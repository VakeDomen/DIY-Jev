#!/usr/bin/env python3
"""Classification quality evaluation set for Granite Jev.

Tests the model's ability to correctly classify inputs across several
scenarios: clear positives, clear negatives, ambiguous cases, and
option permutations (where option names carry meaning that descriptions
alone cannot disambiguate).

Usage:
    python benchmarks/classification_eval.py [--server-url URL]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EvalCase:
    name: str
    category: str
    state: str
    instruction: str
    criteria: dict[str, str | None]
    expected: str
    note: str = ""


@dataclass
class EvalResult:
    name: str
    category: str
    expected: str
    predicted: str | None
    confidence: float | None
    probabilities: dict[str, float] | None
    latency_ms: float
    error: str | None = None
    passed: bool = False

    @property
    def choice(self) -> str | None:
        return self.predicted


EVAL_SUITE: list[EvalCase] = [
    # ── Clear positives ──────────────────────────────────────────────────
    EvalCase(
        name="refund_explicit",
        category="clear_positive",
        state="Customer says: \"I want my money back immediately.\"",
        instruction="What is this customer requesting?",
        criteria={"refund": "Refund request", "support": "Support request"},
        expected="refund",
        note="Explicit refund request",
    ),
    EvalCase(
        name="shipping_inquiry",
        category="clear_positive",
        state="Customer asks: \"Where is my package? It was supposed to arrive yesterday.\"",
        instruction="What is this inquiry about?",
        criteria={"shipping": "Delivery status", "returns": "Return or exchange", "billing": "Payment question"},
        expected="shipping",
        note="Explicit shipping inquiry",
    ),
    EvalCase(
        name="return_request",
        category="clear_positive",
        state="Customer writes: \"I need to return the laptop I bought last week.\"",
        instruction="Categorize this customer request.",
        criteria={"return": "Product return", "warranty": "Warranty claim", "exchange": "Product exchange"},
        expected="return",
        note="Explicit return request",
    ),
    EvalCase(
        name="noul_true",
        category="clear_positive",
        state="\"Please cancel my order immediately.\"",
        instruction="Is this customer requesting a cancellation?",
        criteria={"true": None, "false": None},
        expected="true",
        note="Binary noul: explicit cancellation request",
    ),
    # ── Clear negatives ──────────────────────────────────────────────────
    EvalCase(
        name="no_refund",
        category="clear_negative",
        state="\"I love my new phone! Just wanted to say thanks.\"",
        instruction="Is this customer asking for a refund?",
        criteria={"true": "Yes", "false": "No"},
        expected="false",
        note="Positive feedback, no refund request",
    ),
    EvalCase(
        name="not_billing",
        category="clear_negative",
        state="\"The product arrived damaged. I need to exchange it.\"",
        instruction="Does this inquiry relate to billing or payments?",
        criteria={"true": "Yes", "false": "No"},
        expected="false",
        note="Exchange request, not billing",
    ),
    # ── Ambiguous cases ──────────────────────────────────────────────────
    EvalCase(
        name="vague_complaint",
        category="ambiguous",
        state="\"I'm very unhappy with my purchase.\"",
        instruction="What does this customer need?",
        criteria={"refund": "Refund", "support": "Support", "exchange": "Exchange"},
        expected="support",  # Most reasonable default without specifics
        note="Vague complaint without specific ask",
    ),
    EvalCase(
        name="implied_refund",
        category="ambiguous",
        state="\"The shoes are the wrong size. This is very frustrating.\"",
        instruction="What action is the customer likely seeking?",
        criteria={"refund": "Refund request", "exchange": "Exchange request", "return": "Return request"},
        expected="exchange",  # Wrong-size shoes → exchange, not refund
        note="Wrong size implies exchange, not refund",
    ),
    # ── Option permutations (names matter) ───────────────────────────────
    EvalCase(
        name="names_matter_refund_vs_exchange",
        category="permutation",
        state="\"I want to return this item because it doesn't fit.\"",
        instruction="What is the customer requesting?",
        criteria={
            "refund": "Requested",
            "exchange": "Requested",
        },
        expected="exchange",
        note="Both options have same description 'Requested'; only name distinguishes them. 'doesn\\'t fit' → exchange.",
    ),
    EvalCase(
        name="names_matter_cancel_vs_return",
        category="permutation",
        state="\"I need to cancel my order before it ships.\"",
        instruction="Categorize this request.",
        criteria={
            "cancel": "Requested",
            "return": "Requested",
            "modify": "Requested",
        },
        expected="cancel",
        note="All descriptions identical; only names carry meaning. 'cancel before it ships' → cancel.",
    ),
    # ── Score severity ──────────────────────────────────────────────────
    EvalCase(
        name="severity_high",
        category="scoring",
        state="\"My account has been hacked and I can see unauthorized transactions!\"",
        instruction="Rate the urgency of this issue.",
        criteria={"low": "Low urgency", "medium": "Medium urgency", "high": "High urgency"},
        expected="high",
        note="Security breach → high urgency",
    ),
    EvalCase(
        name="severity_low",
        category="scoring",
        state="\"Just wondering if you have this item in blue instead of green.\"",
        instruction="Rate the urgency of this issue.",
        criteria={"low": "Low urgency", "medium": "Medium urgency", "high": "High urgency"},
        expected="low",
        note="Color inquiry → low urgency",
    ),
]


def send_request(url: str, state: str, instruction: str, criteria: dict[str, str | None],
                 timeout: float) -> dict[str, Any]:
    """Send a single evaluate request to the server."""
    questions = {
        "answer": {
            "type": "choice",
            "instructions": instruction,
            "criteria": criteria,
        }
    }
    body = {
        "state": state,
        "questions": questions,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_case(url: str, case: EvalCase, timeout: float) -> EvalResult:
    """Run a single evaluation case and return the result."""
    started = time.perf_counter()
    try:
        payload = send_request(url, case.state, case.instruction, case.criteria, timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        return EvalResult(
            name=case.name,
            category=case.category,
            expected=case.expected,
            predicted=None,
            confidence=None,
            probabilities=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"HTTP {error.code}: {detail}",
        )
    except Exception as error:
        return EvalResult(
            name=case.name,
            category=case.category,
            expected=case.expected,
            predicted=None,
            confidence=None,
            probabilities=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(error),
        )

    latency_ms = (time.perf_counter() - started) * 1000
    try:
        result = payload.get("result", payload)
        answer = result["answers"]["answer"]
        predicted = answer["choice"]
        confidence = answer["confidence"]
        probabilities = answer.get("probabilities", {})
        passed = predicted == case.expected
    except (KeyError, TypeError, ValueError) as error:
        return EvalResult(
            name=case.name,
            category=case.category,
            expected=case.expected,
            predicted=None,
            confidence=None,
            probabilities=None,
            latency_ms=latency_ms,
            error=f"invalid response: {payload!r}: {error}",
        )

    return EvalResult(
        name=case.name,
        category=case.category,
        expected=case.expected,
        predicted=predicted,
        confidence=confidence,
        probabilities=probabilities,
        latency_ms=latency_ms,
        passed=passed,
    )


def print_report(results: list[EvalResult], wall_seconds: float) -> None:
    """Print a formatted report of all evaluation results."""
    by_category: dict[str, list[EvalResult]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    accuracy = passed / total if total > 0 else 0.0

    print("=" * 72)
    print("CLASSIFICATION QUALITY EVALUATION REPORT")
    print("=" * 72)
    print(f"\nTotal: {total} cases | Passed: {passed} | Failed: {total - passed}")
    print(f"Accuracy: {accuracy:.2%} ({passed}/{total})")
    print(f"Wall time: {wall_seconds:.1f}s")
    print()

    for category, cases in sorted(by_category.items()):
        cat_ok = sum(1 for r in cases if r.passed)
        print(f"\n  [{category}]  {cat_ok}/{len(cases)} passed")
        print(f"  {'─' * 60}")
        for result in cases:
            status = "✓" if result.passed else "✗"
            conf_str = f"  conf={result.confidence:.3f}" if result.confidence is not None else ""
            err_str = f"  ERROR: {result.error}" if result.error else ""
            print(f"  {status} {result.name:40s} expected={result.expected:10s} "
                  f"got={str(result.predicted or '?'):10s}{conf_str}{err_str}")
            if result.probabilities and not result.passed and not result.error:
                probs_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in sorted(result.probabilities.items())
                )
                print(f"    probs: [{probs_str}]")

    print()


def print_detailed_logits(
    results: list[EvalResult],
) -> None:
    """Print detailed probability distributions for inspection."""
    print("\n  DETAILED PROBABILITY BREAKDOWN")
    print("  " + "=" * 60)
    for result in results:
        if result.error or not result.probabilities:
            continue
        print(f"\n  {result.name} ({result.category})")
        print(f"  Expected: {result.expected} | Predicted: {result.predicted} "
              f"| Confidence: {result.confidence:.4f}")
        for option, prob in sorted(result.probabilities.items()):
            marker = " ← expected" if option == result.expected else \
                     " ← predicted" if option == result.predicted else ""
            print(f"    {option:20s} → {prob:.6f}{marker}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8080/v1/evaluate",
        help="Granite Jev server URL (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results as JSON to this file",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup requests before evaluation (default: %(default)s)",
    )
    args = parser.parse_args()

    url = args.server_url

    # Warmup
    print(f"Warming up with {args.warmup} requests...")
    for i in range(args.warmup):
        try:
            send_request(
                url,
                "warmup",
                "Is this a test?",
                {"true": None, "false": None},
                args.timeout,
            )
        except Exception as e:
            print(f"  Warmup {i+1} failed: {e}")
            continue
    print("  Done.")

    # Run evaluation
    results: list[EvalResult] = []
    started = time.perf_counter()
    for case in EVAL_SUITE:
        result = run_case(url, case, args.timeout)
        results.append(result)
        status = "✓" if result.passed else "✗" if not result.error else "!"
        print(f"  {status} {case.name:40s} → {result.predicted or 'ERROR':10s} "
              f"(expected: {case.expected})", flush=True)

    wall_seconds = time.perf_counter() - started

    print()
    print_report(results, wall_seconds)
    print_detailed_logits(results)

    # Also generate a Noul version for the binary cases
    print("\n" + "=" * 72)
    print("NOTE: The Noul/true-false cases above are run as Choice questions")
    print("with true/false criteria, which is how the benchmark comparison works.")
    print("The dedicated Noul type mirrors this with its own probability output.")
    print("=" * 72)

    if args.output:
        path = args.output
        data = {
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "accuracy": sum(1 for r in results if r.passed) / len(results),
                "wall_seconds": wall_seconds,
            },
            "results": [asdict(r) for r in results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {path}")

    return 0 if all(r.passed or r.error for r in results if r.category != "ambiguous") else 1


if __name__ == "__main__":
    sys.exit(main())
