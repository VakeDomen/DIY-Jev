# JevBench public subset — Qwen3.6 35B A3B (user-identified)

Original run: `jevbench-qwen36.zsqaYw`, 2026-09-20,
21:00:18–21:02:08 UTC. The four JSON/JSONL artifacts are unchanged copies.
Raw request/response files remain in the original run directory.

Model identity is not independently verified: the user identifies this as the
A3B run, and the requested model is `qwen3.6-35b-a3b-jev`, but the server
reported `qwen3.5-4b-jev` throughout. The original metadata is preserved.
Confirm the loaded GGUF and correct the server identity before submission.

## Run results

 - Accuracy: 175/231 (75.76%)
 - Median latency: 0.196 s; p95: 1.765 s (serial, localhost)
 - ECE: 0.072; Brier: 0.357 (lower is better)
 - Paraphrase agreement: 30/36 pairs (83.33%)
 - All 231 requests completed with strict-valid distributions; no renormalization.

Dataset hash:
`dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`

This is a public-subset result, not an official JevBench composite score.
Latency measurements do not establish concurrent throughput.
Self-hosted cost is unknown; zero ledger charges do not mean free compute.
