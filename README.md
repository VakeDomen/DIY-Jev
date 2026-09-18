# Granite Jev server

A Jev-compatible constrained-classification HTTP server backed by
`granite-4.2-3b` and llama.cpp. The model is loaded once at startup and all
requests are sent to a resident inference worker. Each question within a request
is evaluated independently (the previous shared-prefix KV-cache optimisation was
removed because it did not produce equivalent results on the Granite
architecture).

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

### Model identity validation

The Cloudflare wrapper accepts an optional `model` field. Currently supported
aliases:

- `typesafe/jev`, `@cf/typesafe/jev`
- `granite-jev`, `granite-jev-0.1.0`

An unrecognised model name returns HTTP 422. The response `model` field always
reflects the actually loaded backend (`granite-jev-0.1.0`).

### Confidence and usage accounting

Confidence is the **maximum softmax probability** among the legal answer tokens.
It is a local per-response statistic, not a calibrated probability of correctness
or an ensemble-derived uncertainty score. The upstream TypeSafe service computes
confidence differently (distribution-derived); this local statistic is clearly
labelled but not directly comparable.

Usage is tracked as:
- `input_tokens`: total tokens across all decoded prompts per question
- `output_tokens`: always 1 per question (the single logit-read step that yields
  the answer distribution)

No output tokens are actually generated through autoregressive decoding. Each
question's answer is read directly from the final prefill logits after a single
forward pass of the prompt.

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

On first use, the script automatically installs `datasets`, `python-chess`, and
the radar plotting dependencies under the gitignored `benchmarks/.deps/`
directory.

The normal benchmark command needs no dataset argument. On its first run it
downloads the source datasets and generates the complete fixture under the
gitignored `benchmarks/data/` directory:

```sh
python3 benchmarks/benchmark.py server \
  --warmup 5
```

The default concurrency is 10, the result is saved to
`benchmarks/results/granite.json`, and the radar is rendered to
`benchmarks/results/radar.png`. Override the request settings with
`--concurrency` or `--output`.

During the run, a dependency-free progress bar on stderr shows completed
requests, throughput, and estimated remaining time.

Create a quick 100-examples-per-axis pilot fixture:

```sh
python3 benchmarks/benchmark.py prepare radar benchmarks/data/radar-pilot.jsonl \
  --limit 100 --seed 0
```

Omit `--limit` for the exact full suite. That is roughly 32,000 questions and
will take a long time against a serialized local server.

```sh
python3 benchmarks/benchmark.py prepare radar benchmarks/data/radar-full.jsonl \
  --seed 0
```

Benchmark the server. The command immediately prints its per-axis deltas from
OpenJev's published NLI-4B numbers and stores the same comparison in the JSON:

```sh
python3 benchmarks/benchmark.py server benchmarks/data/radar-pilot.jsonl \
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
revisions. The exported fixture's SHA-256 fingerprint is recorded in each
server result file.
