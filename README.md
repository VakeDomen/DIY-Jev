# Granite Jev server

A Jev-compatible constrained-classification HTTP server backed by
`granite-4.2-3b` and llama.cpp. The model is loaded once at startup and all
requests are sent to a resident inference worker. Within a request, the shared
state/prompt prefix is encoded once and retained in the KV cache while each
question suffix is scored independently.

This reproduces Jev's public request and response shapes; it is not TypeSafe's
proprietary model or calibration method. Probabilities are a softmax over only
the legal answer labels. Choice/score `confidence` is the largest probability,
and a score is the probability-weighted mean of its rubric indexes.

## Run

```sh
cargo run --release
```

On first start the default GGUF is downloaded to `models/`. Configuration:

- `JEV_BIND_ADDR` — listener address, default `127.0.0.1:8080`
- `JEV_MODEL_PATH` — use an existing GGUF instead of downloading the default
- `JEV_CONTEXT_SIZE` — llama.cpp context size, default `32768`
- `RUST_LOG` — log filter

## API

POST a Jev input directly to `/`, `/v1/evaluate`, or `/ai/run`:

```sh
curl http://127.0.0.1:8080/v1/evaluate \
  -H 'content-type: application/json' \
  -d '{
    "state": "I ordered size 10 shoes but received size 8.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department handles this?",
        "criteria": {
          "returns": "Returns and exchanges",
          "shipping": "Delivery issues",
          "billing": "Charges and refunds"
        }
      },
      "refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?"
      },
      "severity": {
        "type": "score",
        "instructions": "How severe is the problem?",
        "criteria": ["Minor", "Moderate: wrong item", "Major: safety or financial loss"]
      }
    }
  }'
```

The Cloudflare-style body is accepted too:

```json
{
  "model": "typesafe/jev",
  "input": {
    "state": "...",
    "questions": {}
  }
}
```

`GET /health` and `GET /ready` return `ok` after the model and context have
loaded. Inference is serialized because one llama.cpp context owns the KV
cache; HTTP callers may still submit concurrently and are queued.

## Benchmark

`benchmarks/benchmark.py` reproduces the nine-axis suite in
[`AlexWortega/openjev`'s published radar](https://huggingface.co/AlexWortega/openjev):
MMLU, GPQA Diamond, ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag, GSM8K with
4 and 10 choices, and synthetic chess move legality. It reports aggregate and
per-axis top-1/top-3 accuracy, NLL, multiclass Brier score, 10-bin ECE, latency,
and throughput.

Smoke-test a running server:

```sh
python3 benchmarks/benchmark.py server benchmarks/sample.jsonl \
  --output benchmark-results/granite.json
```

Dataset preparation and radar plotting use optional Python dependencies:

```sh
python3 -m pip install datasets matplotlib numpy python-chess
```

Create a quick 100-examples-per-axis pilot fixture:

```sh
python3 benchmarks/benchmark.py prepare radar benchmark-data/radar-pilot.jsonl \
  --limit 100 --seed 0
```

Omit `--limit` for the exact full suite. That is roughly 32,000 questions and
will take a long time against a serialized local server.

```sh
python3 benchmarks/benchmark.py prepare radar benchmark-data/radar-full.jsonl \
  --seed 0
```

Benchmark the server. The command immediately prints its per-axis deltas from
OpenJev's published NLI-4B numbers and stores the same comparison in the JSON:

```sh
python3 benchmarks/benchmark.py server benchmark-data/radar-pilot.jsonl \
  --warmup 5 --output benchmark-results/granite.json

python3 benchmarks/benchmark.py compare \
  benchmark-results/granite.json

python3 benchmarks/benchmark.py radar \
  benchmark-results/granite.json \
  --label "Granite Jev server" \
  --output benchmark-results/radar.png
```

The server benchmark itself uses only Python's standard library. It does not
download or run OpenJev. OpenJev's reference accuracies and sample counts are
copied from its published result files. Model loading and warmup are excluded
from server latency. Pilot-subset deltas are marked as indicative because the
published values cover the complete suite; use the full fixture for the direct
comparison represented by the original chart.

As a sanity check, OpenJev's published full-suite top-1 values (in axis order)
are approximately `47.15%, 27.27%, 76.85%, 59.22%, 58.56%, 31.69%, 39.20%,
17.51%, 24.00%`. Small differences can result from dependency or dataset
revisions; both backends here always consume the same exported fixture, whose
SHA-256 fingerprint is recorded in each result file.
