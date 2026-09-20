# Benchmark analysis

Identical recorded dataset selection.

Historical runs have incomplete server provenance; these are exploratory comparisons.
See ../RESULTS.md for exclusions and validation requirements.

Macro accuracy weights each available task equally; micro accuracy weights every question equally.
Published OpenJev scores are full-suite reference values, not paired predictions.
GSM8K distractor ordering, dataset revisions and subset selection can affect comparability.

| Model | Questions | Tasks | Micro accuracy | Macro accuracy | NLL | ECE | Requests/s | Concurrency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-4-E4B-jev | 32235 | 9 | 60.83% | 54.99% | 0.968 | 0.029 | 26.24 | 30 |
| qwen3.6-35b-a3b-jev | 32235 | 9 | 75.48% | 65.96% | 0.730 | 0.036 | 5.33 | 10 |
| qwen3.8_27b | 32235 | 9 | 75.27% | 65.94% | 0.687 | 0.053 | 2.93 | 10 |
| qwen3_4b | 32235 | 9 | 64.76% | 58.13% | 2.273 | 0.237 | 20.90 | 10 |

## Charts

![accuracy_by_task.png](accuracy_by_task.png)
![delta_vs_openjev.png](delta_vs_openjev.png)
![calibration.png](calibration.png)
![latency_and_throughput.png](latency_and_throughput.png)

## Excluded runs

- results_granite4.2_3b: unresolved model identity
- results_k2_horizon: unresolved model identity

## Sources

- results_gemma-4-E4B-jev/results.json
- results_qwen3.6-35b-a3b-jev/results.json
- results_qwen3.8_27b/results.json
- results_qwen3_4b/results.json
- https://huggingface.co/AlexWortega/openjev/blob/main/results/qwen4b_all.json
- https://huggingface.co/AlexWortega/openjev/blob/main/results/qwen4b_extra_mc.json
