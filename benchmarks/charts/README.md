# Benchmark analysis

Identical recorded dataset selection.

Historical runs have incomplete server provenance; these are exploratory comparisons.
See ../RESULTS.md for exclusions and validation requirements.

Macro accuracy weights each available task equally; micro accuracy weights every question equally.
Published OpenJev scores are full-suite reference values, not paired predictions.
GSM8K distractor ordering, dataset revisions and subset selection can affect comparability.

| Model | Questions | Tasks | Micro accuracy | Macro accuracy | NLL | ECE | Requests/s | Concurrency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 32235 | 9 | 65.56% | 55.36% | 1.235 | 0.337 | 15.27 | 3 |
| Qwen3.6-35B-A3B-UD | 32235 | 9 | 79.63% | 69.55% | 0.695 | 0.183 | 7.64 | 2 |
| systemone/Qwen3.8-27B-UD | 32235 | 9 | 80.79% | 72.89% | 0.566 | 0.089 | 3.65 | 2 |
| systemone/gemma-4-E4B-it-Q4_K_M | 32235 | 9 | 39.78% | 35.50% | 1.378 | 0.118 | 25.38 | 3 |
| systemone/granite-4.2-3b-Q4_K_M | 32235 | 9 | 44.50% | 39.25% | 1.247 | 0.035 | 31.94 | 5 |
| systemone/qwen3-4b-instruct-Q4_K_M | 32235 | 9 | 65.18% | 58.93% | 0.929 | 0.080 | 28.74 | 5 |

## Charts

![accuracy_by_task.png](accuracy_by_task.png)
![delta_vs_openjev.png](delta_vs_openjev.png)
![calibration.png](calibration.png)
![latency_and_throughput.png](latency_and_throughput.png)

## Sources

- results_systemone_Qwen3.5-4B/results.json
- results_systemone_Qwen3.6-35B-A3B-UD/results.json
- results_systemone_Qwen3.8-27B-UD/results.json
- results_systemone_gemma-4-E4B-it-Q4_K_M/results.json
- results_systemone_granite-4.2-3b-Q4_K_M/results.json
- results_systemone_qwen3-4b-instruct-Q4_K_M/results.json
- https://huggingface.co/AlexWortega/openjev/blob/main/results/qwen4b_all.json
- https://huggingface.co/AlexWortega/openjev/blob/main/results/qwen4b_extra_mc.json
