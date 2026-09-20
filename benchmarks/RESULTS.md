# Result status

Keep benchmark results tracked. Historical files have not been retroactively
assigned model hashes, hardware, prompts, or server revisions that were not
recorded at run time.

- `results_granite4.2_3b` and `results_k2_horizon`: identical predictions and
  probability vectors, despite different model labels. Both are excluded from
  default analysis until the served model identity is independently verified.
  Their original JSON and radar images remain for auditability.
- `archived/connection_refused_eval.json`: all requests failed to connect. This
  is a failure log, not an accuracy measurement. Regenerate `eval_results.json`
  with classification_eval.py against a live server when ready.
- Other historical runs have incomplete provenance and are exploratory.

Use `analyze.py --include-unverified` to include the excluded runs explicitly.
Run `analyze.py` after adding/removing results to refresh charts and source paths.
Never claim identical evaluation merely from matching sample counts. Retain the
fixture hash, revision manifest, server commit, model hash, prompts, quantization,
hardware, batching settings and client concurrency with a publishable run.

Copy `provenance.example.json`, replace the placeholders with the actual server
values, and pass it to `benchmark.py server --provenance <your-file.json>`.
This metadata is explicitly user-supplied, not automatically verified against
the server. Never store tokens or credentials in the provenance file.

For repeatable fixture generation, retain the `.manifest.json` sidecar alongside
the fixture. Re-running preparation at the same output path reuses its recorded
dataset commit IDs. Historical fixtures without a manifest remain unpinned;
their content hashes identify the files but cannot recover the source revisions.
