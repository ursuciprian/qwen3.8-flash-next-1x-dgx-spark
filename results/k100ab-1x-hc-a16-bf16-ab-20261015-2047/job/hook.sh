#!/usr/bin/env bash
# k100 Thunderdome hook (copy of k90), run after each boot's probes: bash hook.sh <host> <boot dir>
#   cold     cold prefill 16K / 64K / 128K, 3 prompts each (cold.py seed "hook": the same prompts in every boot, so
#            control and arm boots pair; nonce-led, no prefix-cache hits), from a corpus range no probe reads
#   facts    from the live serve log: b12x plans measured (must be 0), GPU KV cache size, AOT load line
#   profile  pass-1 boots only (<who>-p1): torch-profiler window over one more cold 16K prompt; the trace is tarred out
#            of the container to <boot dir>/hook/prof-k100/ (excluded from the results PR by the poster)
set -u
h=$1; b=$2; d=$b/hook; mkdir -p "$d"
K=$HOME/GEN-AI/backlog/k100; CORPUS=$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/corpus-code.txt
H1=$(. "$HOME/.config/spark-pair.env"; echo "$SPARK_HEAD_IP")
x() { if [ "$1" = "$H1" ]; then bash -c "$2"; else ssh -n -o ConnectTimeout=10 "$1" "$2"; fi; }
timeout -k 30 2400 python3 "$K/cold.py" run --base "http://$h:8000" --corpus "$CORPUS" --lens 16384,65536,131072 --reps 3 \
  --seed hook --cskip 0 --out "$d/cold.json" > "$d/cold.txt" 2>&1; rc=$?
c=$(x "$h" "docker ps --format '{{.Names}}' | grep sparkrun | head -1")
x "$h" "docker exec $c cat /tmp/sparkrun_serve.log" > "$d/serve-at-hook.log" 2>/dev/null
{ echo "plans_measured $(grep -ohE '\b[0-9]+ measured' "$d/serve-at-hook.log" | awk '{s+=$1} END {print s+0}')"
  echo "kv $(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$d/serve-at-hook.log")"
  echo "aot $(grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' "$d/serve-at-hook.log")"
  echo "mnbt $(grep -m1 -oE 'max.num.batched.tokens[^0-9]{1,6}[0-9]+' "$d/serve-at-hook.log")"; } > "$d/facts.txt"
rm -f "$d/serve-at-hook.log"
case "$(basename "$b")" in
  *-p1)
    curl -s -m 60 -X POST "http://$h:8000/start_profile" > "$d/start_profile.txt" 2>&1
    timeout -k 30 600 python3 "$K/cold.py" run --base "http://$h:8000" --corpus "$CORPUS" --lens 16384 --reps 1 --seed "prof-$(basename "$b")" \
      --cskip 3300000 --out "$d/cold-prof.json" > "$d/cold-prof.txt" 2>&1
    curl -s -m 900 -X POST "http://$h:8000/stop_profile" > "$d/stop_profile.txt" 2>&1; sleep 20
    x "$h" "docker exec $c sh -c 'cd /cache/runtime && tar -cz prof-k100 && rm -rf prof-k100'" > "$d/trace.tgz" 2>/dev/null
    ( cd "$d" && tar -xzf trace.tgz && rm -f trace.tgz ) ;;
esac
tail -4 "$d/cold.txt"; cat "$d/facts.txt"; exit $rc
