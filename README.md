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
