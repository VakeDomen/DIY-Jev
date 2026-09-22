#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 benchmarks/benchmark.py radar \
    benchmarks/results_systemone_qwen3-4b-instruct-Q4_K_M/results.json \
    benchmarks/results_systemone_Qwen3.5-4B/results.json \
    benchmarks/results_systemone_Qwen3.8-27B-UD/results.json \
    benchmarks/results_systemone_Qwen3.6-35B-A3B-UD/results.json \
    benchmarks/results_systemone_granite-4.2-3b-Q4_K_M/results.json \
    benchmarks/results_systemone_gemma-4-E4B-it-Q4_K_M/results.json \
    benchmarks/results_systemone_Phi-4-mini-instruct-Q4_K_M/results.json \
    --label "Qwen3 4B" \
    --label "Qwen3.5 4B" \
    --label "Qwen3.8 27B" \
    --label "Qwen3.6 35B-A3B" \
    --label "Granite 4.2 3B" \
    --label "Gemma 4B" \
    --label "Phi4 Mini" \
    --output benchmarks/charts/combined-radar.png \
    --title "DIY Jev — Model Comparison"
