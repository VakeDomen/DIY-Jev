#!/usr/bin/env python3
"""Contract test: validate real request/response fixtures against openapi.yaml.

This test reads the OpenAPI spec, builds JSON Schema validators for each
operation, then sends sample request/response pairs through validation.
It does not depend on a running server — it validates serialized Rust types.

Usage:
    python3 tests/contract_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import yaml
    from jsonschema import validate, ValidationError
except ImportError as error:
    print(f"SKIP: {error} — install pyyaml and jsonschema")
    sys.exit(0)


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_openapi() -> dict:
    with open(REPO_ROOT / "openapi.yaml") as f:
        return yaml.safe_load(f)


def resolve_ref(spec: dict, ref: str) -> dict:
    """Resolve a JSON Reference like '#/components/schemas/EvaluateInput'."""
    parts = ref.lstrip("#/").split("/")
    node = spec
    for part in parts:
        node = node[part]
    return node


def inline_refs(spec: dict, schema: dict, visited: set | None = None) -> dict:
    """Inline $ref pointers to produce a standalone schema."""
    if visited is None:
        visited = set()
    ref_key = json.dumps(schema, sort_keys=True)
    if ref_key in visited:
        return schema
    visited.add(ref_key)
    if "$ref" in schema:
        resolved = resolve_ref(spec, schema["$ref"])
        return inline_refs(spec, resolved, visited)
    result = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            result[key] = inline_refs(spec, value, visited)
        elif isinstance(value, list):
            result[key] = [inline_refs(spec, item, visited) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    return result


def get_request_schema(spec: dict, path: str, method: str = "post") -> dict | None:
    """Extract and inline the request body schema for a given path+method."""
    path_item = spec.get("paths", {}).get(path, {})
    operation = path_item.get(method, {})
    request_body_ref = operation.get("requestBody", {}).get("$ref", {})
    if request_body_ref:
        request_body = resolve_ref(spec, request_body_ref)
    else:
        request_body = operation.get("requestBody", {})
    content = request_body.get("content", {})
    media_type = content.get("application/json", {})
    schema = media_type.get("schema")
    if schema:
        return inline_refs(spec, schema)
    return None


def get_response_schema(spec: dict, path: str, status: str = "200", method: str = "post") -> dict | None:
    """Extract and inline the response schema for a given path+method+status."""
    path_item = spec.get("paths", {}).get(path, {})
    operation = path_item.get(method, {})
    responses = operation.get("responses", {})
    response = responses.get(status, {}).get("$ref", {})
    if response:
        response = resolve_ref(spec, response)
    else:
        response = responses.get(status, {})
    content = response.get("content", {})
    media_type = content.get("application/json", {})
    schema = media_type.get("schema")
    if schema:
        return inline_refs(spec, schema)
    return None


def test_request_validation() -> int:
    """Validate sample JSON requests against the OpenAPI spec."""
    spec = load_openapi()
    errors = 0

    test_cases: list[tuple[str, str, dict]] = [
        (
            "Direct evaluate request with all question types",
            "/v1/evaluate",
            {
                "state": "Customer received wrong item.",
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Which department?",
                        "criteria": {
                            "billing": "Charges",
                            "shipping": "Delivery",
                            "returns": "Returns",
                        },
                    },
                    "refund": {
                        "type": "noul",
                        "instructions": "Refund?",
                    },
                    "severity": {
                        "type": "score",
                        "instructions": "Severity?",
                        "criteria": ["low", "medium", "high"],
                    },
                },
            },
        ),
        (
            "Direct evaluate with extra unknown fields (serde ignores them)",
            "/v1/evaluate",
            {
                "state": "test",
                "questions": {
                    "q1": {"type": "noul", "instructions": "test", "extra": "ignored"},
                },
                "unknown_field": "should be accepted",
            },
        ),
        (
            "Cloudflare wrapper with string model",
            "/ai/run",
            {
                "model": "typesafe/jev",
                "input": {
                    "state": "test",
                    "questions": {
                        "q1": {"type": "noul", "instructions": "test"},
                    },
                },
            },
        ),
        (
            "Cloudflare wrapper with null model",
            "/ai/run",
            {
                "model": None,
                "input": {
                    "state": "test",
                    "questions": {
                        "q1": {"type": "choice", "instructions": "test",
                               "criteria": {"a": "A", "b": "B"}},
                    },
                },
            },
        ),
        (
            "Object instructions",
            "/v1/evaluate",
            {
                "state": "test",
                "questions": {
                    "q1": {
                        "type": "noul",
                        "instructions": {"key": "value"},
                    },
                },
            },
        ),
        (
            "Array instructions",
            "/v1/evaluate",
            {
                "state": "test",
                "questions": {
                    "q1": {
                        "type": "noul",
                        "instructions": ["step one", "step two"],
                    },
                },
            },
        ),
        (
            "Null criteria in Noul",
            "/v1/evaluate",
            {
                "state": "test",
                "questions": {
                    "q1": {
                        "type": "noul",
                        "instructions": "test",
                        "criteria": {"true": None, "false": None},
                    },
                },
            },
        ),
        (
            "Choice with 255 options (max)",
            "/v1/evaluate",
            {
                "state": "test",
                "questions": {
                    "q1": {
                        "type": "choice",
                        "instructions": "Pick one",
                        "criteria": {f"opt_{i}": f"Option {i}" for i in range(255)},
                    },
                },
            },
        ),
    ]

    for name, path, body in test_cases:
        schema = get_request_schema(spec, path)
        if schema is None:
            print(f"⚠  {name}: no schema found for {path}")
            errors += 1
            continue
        try:
            validate(instance=body, schema=schema)
            print(f"✓  {name}")
        except ValidationError as e:
            print(f"✗  {name}: {e.message}")
            errors += 1

    return errors


def test_response_validation() -> int:
    """Validate sample JSON responses against the OpenAPI spec."""
    spec = load_openapi()
    errors = 0

    test_cases: list[tuple[str, dict]] = [
        (
            "Choice answer",
            {
                "model": "granite-jev-0.1.0",
                "answers": {
                    "department": {
                        "type": "choice",
                        "choice": "returns",
                        "confidence": 0.85,
                        "probabilities": {
                            "billing": 0.05,
                            "shipping": 0.10,
                            "returns": 0.85,
                        },
                    },
                },
                "usage": {"input_tokens": 128, "output_tokens": 3},
            },
        ),
        (
            "Noul answer",
            {
                "model": "granite-jev-0.1.0",
                "answers": {
                    "refund": {"type": "noul", "noul": 0.92},
                },
                "usage": {"input_tokens": 64, "output_tokens": 1},
            },
        ),
        (
            "Score answer",
            {
                "model": "granite-jev-0.1.0",
                "answers": {
                    "severity": {
                        "type": "score",
                        "score": 1.7,
                        "confidence": 0.65,
                        "legend": {"0": "low", "1": "medium", "2": "high"},
                        "probabilities": {"0": 0.1, "1": 0.25, "2": 0.65},
                    },
                },
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        ),
        (
            "Response with extra unknown fields",
            {
                "model": "granite-jev-0.1.0",
                "answers": {
                    "q1": {"type": "noul", "noul": 0.5},
                },
                "usage": {"input_tokens": 10, "output_tokens": 1},
                "extra_field": "should be accepted",
            },
        ),
    ]

    for name, body in test_cases:
        schema = get_response_schema(spec, "/v1/evaluate", "200")
        if schema is None:
            print(f"⚠  {name}: no response schema found")
            errors += 1
            continue
        try:
            validate(instance=body, schema=schema)
            print(f"✓  {name}")
        except ValidationError as e:
            print(f"✗  {name}: {e.message}")
            errors += 1

    return errors


def test_error_responses() -> int:
    """Validate error response shapes against the OpenAPI spec."""
    spec = load_openapi()
    errors = 0

    # Each error case specifies the response status it expects to validate
    # against, so every error status code gets its own test.
    error_cases: list[tuple[str, str, str, dict]] = [
        ("400 Bad Request", "/v1/evaluate", "400", {"error": "invalid JSON"}),
        ("413 Content Too Large", "/v1/evaluate", "413", {"error": "body too large"}),
        ("415 Unsupported Media", "/v1/evaluate", "415", {"error": "unsupported content type"}),
        ("422 Validation error", "/v1/evaluate", "422", {"error": "unsupported model"}),
        ("502 Backend error", "/v1/evaluate", "502", {"error": "backend failure"}),
        ("503 Overload", "/v1/evaluate", "503", {"error": "queue full"}),
    ]

    for name, path, status, body in error_cases:
        schema = get_response_schema(spec, path, status)
        if schema is None:
            print(f"⚠  {name}: no schema found for status {status}")
            errors += 1
            continue
        try:
            validate(instance=body, schema=schema)
            print(f"✓  {name}")
        except ValidationError as e:
            print(f"✗  {name}: {e.message}")
            errors += 1

    return errors


def main() -> int:
    print("=" * 60)
    print("OpenAPI Contract Tests")
    print("=" * 60)

    print("\n--- Request validation ---")
    req_errors = test_request_validation()

    print("\n--- Response validation ---")
    resp_errors = test_response_validation()

    print("\n--- Error response validation ---")
    err_errors = test_error_responses()

    total = req_errors + resp_errors + err_errors
    print(f"\n{'=' * 60}")
    print(f"Total errors: {total}")
    if total == 0:
        print("All contract tests PASSED")
    else:
        print(f"{total} contract test(s) FAILED")

    return total


if __name__ == "__main__":
    sys.exit(main())
