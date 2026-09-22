# JevBench public subset — Qwen3.5 4B (user-identified)

Original run: `jevbench-qwen36.zqKmw6`, 2026-09-20,
21:28:34–21:29:25 UTC. The four JSON/JSONL artifacts are unchanged copies.
The original run, including raw request/response files, remains in place.

The user identifies this as the Qwen3.5 4B run. The server reports
`qwen3.5-4b-jev`, but the requested model and run label still refer to
`qwen3.6-35b-a3b-jev`. Original metadata is preserved; no GGUF hash was
recorded to independently verify the loaded weights.

## Run results

- Accuracy: 139/231 (60.17%)
- Easy: 39/48 (81.25%)
- Original: 49/72 (68.06%)
- Hard: 51/111 (45.95%)
- Median latency: 0.087 s; p95: 0.804 s (serial, localhost)
- ECE: 0.269; Brier: 0.634 (lower is better)
- All 231 requests completed with strict-valid distributions; no renormalization.

Dataset hash:
`dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`

This is a public-subset result, not an official JevBench composite score.
Self-hosted cost is unknown; zero ledger charges do not mean free compute.
