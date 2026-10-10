#!/usr/bin/env bash
# k78 Thunderdome hook (copy of k74/hook.sh): cold prefill 64K and 128K, 3 fresh prompts each (cold.py, seed "hook": the same prompts in
# every boot, so control and arm boots are paired), run after each boot's probes. Usage: bash hook.sh <host> <boot dir>
set -u
h=$1; d=$2/hook; mkdir -p "$d"
K=$HOME/GEN-AI/backlog/k78; CORPUS=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/corpus-code.txt
timeout -k 30 1800 python3 "$K/cold.py" run --base "http://$h:8000" --corpus "$CORPUS" --lens 65536,131072 --reps 3 --seed hook \
  --out "$d/cold.json" > "$d/cold.txt" 2>&1; rc=$?
tail -2 "$d/cold.txt"; exit $rc
