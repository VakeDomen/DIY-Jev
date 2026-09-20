# JevBench public subset — Qwen3 4B (user-reported, unverified)

Original run: `jevbench-qwen36.BUt0oi`, 2026-09-20,
21:25:41–21:26:20 UTC. The four JSON/JSONL artifacts are unchanged copies.
The original run, including raw request/response files, remains in place.

The user identifies this as a new Qwen3 run. However, the manifest requests
`qwen3.6-35b-a3b-jev` and the server reports `qwen3.5-4b-jev`.
All 231 predictions and returned probability distributions exactly match
the earlier run `jevbench-qwen36.cOm5UD`. The user subsequently confirmed
that both runs used Qwen3, not Qwen3.5. The earlier run is now archived in
`../jevbench-earlier/`. Attribution is based on the user's correction,
not an independently verified GGUF hash.
Original metadata has not been rewritten.

## Run results

- Accuracy: 132/231 (57.14%)
- Easy: 41/48 (85.42%)
- Original: 41/72 (56.94%)
- Hard: 50/111 (45.05%)
- Median latency: 0.062 s; p95: 0.608 s (serial, localhost)
- ECE: 0.164; Brier: 0.576
- All 231 requests completed with strict-valid distributions; no renormalization.

Dataset hash:
`dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`

This is a public-subset result, not an official JevBench composite score.
Self-hosted cost is unknown; zero ledger charges do not mean free compute.
