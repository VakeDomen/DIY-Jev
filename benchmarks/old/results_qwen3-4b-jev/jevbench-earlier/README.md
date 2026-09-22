# JevBench public subset — Qwen3 4B (earlier run, corrected attribution)

Original run: `jevbench-qwen36.cOm5UD`, 2026-09-20.
The four JSON/JSONL artifacts are unchanged copies of the original run.
Raw request/response files remain in the original run directory.

The user confirmed this run used Qwen3 4B, not Qwen3.5. It was initially
misfiled based on the stale server identity `qwen3.5-4b-jev`; the requested
model also incorrectly refers to Qwen3.6. Original metadata is preserved.
Attribution is user-provided, not independently verified by GGUF hash.

- Accuracy: 132/231 (57.1%)
- Easy: 41/48; original: 41/72; hard: 50/111
- Median latency: 0.056 s; p95: 0.602 s (serial, localhost)
- ECE: 0.164; Brier: 0.576
- All 231 requests completed with strict-valid distributions.

This is a public-subset result, not an official JevBench composite score.
Self-hosted cost is unknown; a zero ledger charge is not a zero compute cost.
The manifest records the server-reported identity, not a verified GGUF hash.
