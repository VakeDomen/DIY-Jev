# DIY Jev server

A Jev-compatible constrained-classification HTTP server backed by llama.cpp.
The model is loaded once at startup and all requests are sent to a resident
inference worker. Each question within a request is evaluated independently
(the previous shared-prefix KV-cache optimisation was removed because it did
not produce equivalent results on some architectures).

Any GGUF model can be used — set `JEV_HF_REPO` and `JEV_HF_FILENAME` to
auto-download from Hugging Face, or point `JEV_MODEL_PATH` at an existing
GGUF file. The default is Granite 4.2 3B.

This reproduces Jev's public request and response shapes; it is not TypeSafe's
proprietary model or calibration method. Probabilities are a softmax over only
the legal answer labels. Choice/score `confidence` is the largest probability,
and a score is the probability-weighted mean of its rubric indexes.

## Run

Enable experimental cross-request batching with `JEV_REQUEST_BATCH_SIZE=10`.
Queued single-question requests use independent candidate sequence IDs.
`JEV_REQUEST_BATCH_SIZE` defaults to `1` (the individual baseline), and
`JEV_REQUEST_BATCH_WAIT_MS` defaults to `2`.
Requests are grouped into waves constrained by `JEV_N_SEQ_MAX` and the context
token budget; the default 16 sequences fits four 4-choice requests per wave.
Multi-question requests retain individual evaluation. Disconnected queued
callers are skipped. A decode failure fails the affected wave, while validation
errors are handled per request. This is bounded microbatching, not continuous
admission of new requests during an active wave.

Batch shape can affect model scores: the local Qwen Q4 CPU test showed a
Score probability shift of roughly 0.28 with ordinary microbatch settings.
Batching is therefore opt-in; validate accuracy on your model/backend before
using it for benchmark comparisons. No GPU throughput gain has been measured.

Compare throughput and p95 latency with request batch sizes 1, 2, 4 and 10.
Increasing `JEV_N_SEQ_MAX` permits more candidates per wave but consumes more
model-dependent memory. Enable `RUST_LOG=diy_jev=debug` to see wave sizes.
The opt-in CPU-capable sequence-isolation test fixes physical microbatch size
to 1 and checks probabilities within 0.001. Set `JEV_TEST_UBATCH=32` to check
batch-dependent numerical drift as well (this currently fails on Qwen Q4 CPU).
Run the isolation test with:

```sh
JEV_TEST_MODEL=./models/qwen3-4b-instruct-Q4_K_M.gguf \
  cargo test --release --no-default-features --test cross_request_test
```

Inference now uses raw tagged prompts, without applying or requiring a model
chat template. Instructions precede escaped `<state>`, `<question>`, and
`<candidate>` data, followed by an open `<answer>` tag and a newline to prevent
the tokenizer from merging the tag with the answer. Noul omits
the candidate. The model's next-token logits for lowercase `true` and `false`
are scored at that boundary; no answer text or closing tag is generated.
Startup verifies both continuations are single tokens without changing the
prompt's token prefix. Existing candidate batching and scoring are retained.

```sh
cargo run --release
```

On first start the default GGUF is downloaded to `models/`. Configuration:

- `JEV_BIND_ADDR` — listener address, default `127.0.0.1:8080`
- `JEV_MODEL_PATH` — use an existing GGUF instead of downloading the default
- `JEV_HF_REPO` — Hugging Face repo (e.g. `unsloth/Qwen3.6-35B-A3B-GGUF`)
- `JEV_HF_FILENAME` — GGUF filename in that repo
- `JEV_MODEL_IDENTITY` — model string returned in API responses, default `diy-jev-0.1.0`
- `JEV_MODEL_ALIASES` — comma-separated accepted model aliases
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

Additional aliases can be configured via `JEV_MODEL_ALIASES`. An unrecognised
model name returns HTTP 422. The response `model` field always reflects the
actually loaded backend (default `diy-jev-0.1.0`, overridable via
`JEV_MODEL_IDENTITY`).

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
  --output benchmark-results/results.json
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

The default concurrency is 10. Results and the radar are saved to
`benchmarks/results_<model name>/results.json` and `radar.png`. The model name
comes from the server response; use `--model-name qwen3_4b` to override it.
Names are sanitized for filesystem paths. A custom `--output` takes precedence,
and the radar is saved beside that JSON file.

Analyze all saved model runs and generate charts:

```sh
python3 benchmarks/analyze.py
```

This discovers `benchmarks/results_*/*.json` and writes task accuracy and
published-baseline comparison heatmaps, calibration curves, confidence
distributions, latency CDFs and a throughput/accuracy plot into
`benchmarks/charts/`. That directory also contains a summary `README.md` and
`task_metrics.csv`. Supply explicit JSON paths to analyze selected runs.
The report checks dataset-selection fingerprints; speed comparisons reflect
the recorded hardware and concurrency, which may differ between runs.

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
  --warmup 5 --output benchmark-results/results.json

python3 benchmarks/benchmark.py compare \
  benchmark-results/results.json

python3 benchmarks/benchmark.py radar \
  benchmark-results/results.json \
  --label "DIY Jev server" \
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
