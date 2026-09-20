#!/usr/bin/env python3
"""HTTP integration test: validates Axum error responses match the OpenAPI spec.

Starts the DIY Jev server (or connects to an existing one), sends requests
that trigger each error status code, and verifies the JSON error envelope
matches the expected schema.

Usage:
    python3 tests/http_test.py [--server-url URL]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_server(url: str, timeout: float = 30.0) -> bool:
    """Check if the server is running and healthy."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def send_raw(url: str, body: bytes | None, content_type: str = "application/json",
             method: str = "POST", timeout: float = 30.0) -> tuple[int, bytes | None]:
    """Send an HTTP request and return (status_code, response_body_or_None)."""
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("content-type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        return 0, str(e).encode()


def test_json_error_envelope(url: str) -> int:
    """Test that Axum errors return a JSON body with an 'error' field.

    Every error response MUST:
      - Return the expected HTTP status code.
      - Return a JSON body matching the `Error` schema: {"error": "..."}.

    Axum's default 415 response (without our envelope) is NOT accepted.
    Any 200 for invalid input is a FAILURE.
    """
    errors = 0
    base = f"{url}/v1/evaluate"

    test_cases: list[tuple[str, bytes | None, str, int, str]] = [
        (
            "non-JSON body with wrong content-type",
            b"not json",
            "text/plain",
            415,
            "{",
        ),
        (
            "invalid JSON body",
            b"not json",
            "application/json",
            400,
            "{",
        ),
        (
            "empty body",
            b"",
            "application/json",
            400,
            "{",
        ),
        (
            "valid JSON but no state",
            json.dumps({"questions": {"q1": {"type": "noul", "instructions": "test"}}}).encode(),
            "application/json",
            422,
            "{",
        ),
        (
            "wrong content-type (text/xml)",
            json.dumps({"state": "test", "questions": {"q1": {"type": "noul", "instructions": "test"}}}).encode(),
            "text/xml",
            415,
            "{",
        ),
    ]

    for name, body, content_type, expected_status, _expected_hint in test_cases:
        status, data = send_raw(base, body, content_type)

        if status == 0:
            print(f"✗  {name} → connection refused (server not running)")
            errors += 1
            continue

        # The response must be valid JSON with an "error" field.
        try:
            parsed = json.loads(data) if data else {}
        except (json.JSONDecodeError, TypeError):
            parsed = None

        if status != expected_status:
            print(f"✗  {name} → expected HTTP {expected_status}, got {status}")
            errors += 1
            continue

        if parsed is None or not isinstance(parsed, dict) or "error" not in parsed:
            print(f"✗  {name} → HTTP {status} but body is not a JSON {{error: ...}}: {data!r}")
            errors += 1
            continue

        print(f"✓  {name} → {status} with JSON error envelope")

    return errors


def test_model_validation(url: str) -> int:
    """Test that invalid model names are rejected with proper error envelope."""
    errors = 0
    base = f"{url}/v1/evaluate"

    body = json.dumps({
        "model": "nonexistent/model",
        "input": {
            "state": "test",
            "questions": {"q1": {"type": "noul", "instructions": "test"}},
        },
    }).encode()

    status, data = send_raw(base, body)
    if status == 0:
        print(f"✗  Invalid model → connection refused")
        return errors + 1

    try:
        parsed = json.loads(data) if data else {}
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if status != 422:
        print(f"✗  Invalid model → expected HTTP 422, got {status}")
        errors += 1
    elif parsed is None or not isinstance(parsed, dict) or "error" not in parsed:
        print(f"✗  Invalid model → HTTP 422 but body is not a JSON {{error: ...}}: {data!r}")
        errors += 1
    elif "unsupported model" not in parsed.get("error", "").lower():
        print(f"✗  Invalid model → HTTP 422 but error message does not mention 'unsupported model': {parsed['error']!r}")
        errors += 1
    else:
        print(f"✓  Invalid model → 422 with JSON error envelope")

    return errors


def test_health_and_readiness(url: str) -> int:
    """Test that health and readiness endpoints work."""
    errors = 0

    for endpoint in ("/health", "/ready"):
        status, data = send_raw(f"{url}{endpoint}", None, method="GET")
        if status == 200:
            body = data.decode() if data else ""
            if body == "ok":
                print(f"✓  {endpoint} → 200 ok")
            else:
                print(f"~  {endpoint} → 200 but body={body!r}")
        else:
            print(f"✗  {endpoint} → {status}")
            errors += 1

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8088")
    parser.add_argument("--start-server", action="store_true",
                        help="Start the server binary before testing")
    parser.add_argument("--server-bin", type=Path,
                        default=REPO_ROOT / "target/release/diy-jev")
    args = parser.parse_args()

    url = args.server_url.rstrip("/")

    if args.start_server:
        if not args.server_bin.exists():
            print(f"Server binary not found: {args.server_bin}")
            print("Build with: cargo build --release")
            return 1
        print(f"Starting server: {args.server_bin}")
        env = {"CUDA_VISIBLE_DEVICES": "", "JEV_BIND_ADDR": "127.0.0.1:8088"}
        proc = subprocess.Popen(
            [str(args.server_bin)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**dict(subprocess.os.environ), **env},
        )
        try:
            # Wait for the server to start
            for _ in range(30):
                if check_server(url):
                    print("Server ready.")
                    break
                time.sleep(1)
            else:
                print("Server did not start within 30 seconds")
                proc.kill()
                return 1

            total = run_tests(url)
        finally:
            proc.kill()
            proc.wait()
    else:
        if not check_server(url):
            print(f"Server not reachable at {url}")
            print("Start it or use --start-server to launch automatically")
            return 1
        total = run_tests(url)

    return total


def run_tests(url: str) -> int:
    print("=" * 60)
    print("HTTP Integration Tests")
    print("=" * 60)

    print("\n--- Health / Readiness ---")
    total = test_health_and_readiness(url)

    print("\n--- JSON error envelope ---")
    total += test_json_error_envelope(url)

    print("\n--- Model validation ---")
    total += test_model_validation(url)

    print(f"\n{'=' * 60}")
    print(f"Total errors: {total}")
    if total == 0:
        print("All HTTP tests PASSED")
    else:
        print(f"{total} HTTP test(s) FAILED")

    return total


if __name__ == "__main__":
    sys.exit(main())
