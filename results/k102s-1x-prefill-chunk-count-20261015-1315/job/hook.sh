#!/usr/bin/env bash
# k102 Thunderdome hook, run after each boot's probes: bash hook.sh <host> <boot dir>
#   cold     cold prefill 16K / 64K / 128K, 3 prompts each (k101/cold.py seed "hook": the same prompts in every boot;
#            nonce-led, no prefix-cache hits)
#   facts    from the live serve log: b12x plans measured (must be 0), GPU KV cache size, AOT load line, budget, and the
#            logged chunk list of the last cold 16K / 64K / 128K prompt (arm only: the control does not log splits)
set -u
h=$1; b=$2; d=$b/hook; mkdir -p "$d"
K=$HOME/GEN-AI/backlog/k102; C=$HOME/GEN-AI/backlog/k101; CORPUS=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/corpus-code.txt
H1=$(. "$HOME/.config/spark-pair.env"; echo "$SPARK_HEAD_IP")
x() { if [ "$1" = "$H1" ]; then bash -c "$2"; else ssh -n -o ConnectTimeout=10 "$1" "$2"; fi; }
timeout -k 30 2400 python3 "$C/cold.py" run --base "http://$h:8000" --corpus "$CORPUS" --lens 16384,65536,131072 --reps 3 \
  --seed hook --cskip 0 --out "$d/cold.json" > "$d/cold.txt" 2>&1; rc=$?
c=$(x "$h" "docker ps --format '{{.Names}}' | grep sparkrun | head -1")
x "$h" "docker exec $c cat /tmp/sparkrun_serve.log" > "$d/serve-at-hook.log" 2>/dev/null
{ echo "plans_measured $(grep -ohE '\b[0-9]+ measured' "$d/serve-at-hook.log" | awk '{s+=$1} END {print s+0}')"
  echo "kv $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$d/serve-at-hook.log")"
  echo "aot $(grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' "$d/serve-at-hook.log")"
  echo "mnbt $(grep -m1 -oE 'max.num.batched.tokens[^0-9]{1,6}[0-9]+' "$d/serve-at-hook.log")"
  for n in 16384 65536 131072; do echo "chunks$n $(python3 "$K/splits.py" steps "$d/serve-at-hook.log" $n)"; done; } > "$d/facts.txt"
python3 "$K/splits.py" table "$d/serve-at-hook.log" > "$d/splits.txt" 2>&1
rm -f "$d/serve-at-hook.log"
tail -4 "$d/cold.txt"; cat "$d/facts.txt"; exit $rc
