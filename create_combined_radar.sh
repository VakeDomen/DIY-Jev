#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 benchmarks/benchmark.py radar \
    benchmarks/results_qwen3_4b/results.json \
    benchmarks/results_qwen3.8_27b/results.json \
    benchmarks/results_qwen3.6-35b-a3b-jev/results.json \
    benchmarks/results_granite4.2_3b/results.json \
    benchmarks/results_gemma-4-E4B-jev/results.json \
    --label "Qwen3 4B" \
    --label "Qwen3.8 27B" \
    --label "Qwen3.6 35B-A3B" \
    --label "Granite 4.2 3B" \
    --label "Gemma 4B" \
    --output benchmarks/charts/combined-radar.png \
    --title "DIY Jev — Model Comparison"
