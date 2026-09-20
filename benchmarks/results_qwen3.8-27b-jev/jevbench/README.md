# JevBench public subset — Qwen3.8 27B (user-identified)

Original run: `jevbench-qwen36.XPPZBd`, 2026-09-20,
21:06:14–21:10:25 UTC.
The four JSON/JSONL artifacts are unchanged copies of the original run.
Raw request/response files remain in the original run directory.

This timing rerun replaces `jevbench-run.FytHPF`, which the user reports was
measured while other workloads were running. That earlier run remains in its
original directory. All 231 predictions and aggregate accuracy/calibration
metrics are unchanged.

Model identity is user-confirmed, not independently verified: the manifest
requests `qwen3.6-35b-a3b-jev` and the server reports `qwen3.5-4b-jev`, despite
the user identifying the loaded model as 27B. Original metadata is preserved;
confirm the GGUF and correct the server identity before submission.

- Accuracy: 195/231 (84.4%)
- Easy: 48/48; original: 66/72; hard: 81/111
- Median latency: 0.448 s; p95: 3.946 s (serial, localhost)
- ECE: 0.166; Brier: 0.269
- All 231 requests completed with strict-valid distributions.

This is a public-subset result, not an official JevBench composite score.
Self-hosted cost is unknown; a zero ledger charge is not a zero compute cost.
The manifest records the server-reported identity, not a verified GGUF hash.
