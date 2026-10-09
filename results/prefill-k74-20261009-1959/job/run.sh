#!/usr/bin/env bash
# k74 (2026-10-08, 1x repo issue): long-context prefill on the single Spark (v3e). Headline cell: 128K.
#   phase 1  measure, both Sparks in parallel, no arm:
#            dgx-01  solo boot of the registry recipe @qwen38-flashnext-1x/qwen3.8-flash-next-1x-dgx-spark (v3e);
#                    cold prefill 16K/64K/128K (cold.py: fresh prompts, 3 repeats, prompt tokens / mean TTFT), then
#                    llm-inference-bench --standalone-prefill --prefill-only 16k,64k,128k (the README definition),
#                    so both definitions are on record from one boot
#            dgx-02  v3e + torch profiler (own VLLM_CACHE_ROOT, k64 pattern): one window over a cold 16K request and
#                    one over a cold 128K request; prof_summary.py per window (what grows with length)
#   phase 2  Thunderdome (scripts/thunderdome.sh), one config-only arm on both Sparks: v3e with
#            max_num_batched_tokens 8192 -> 16384 (own XDG_CACHE_HOME + VLLM_CACHE_ROOT, bake first: compile ranges and
#            b12x plans change), control = v3e; hook.sh runs cold 64K/128K x 3 on every boot (paired prompts);
#            normal verdict, split gate on a PROMOTE (thunderdome default); thunderdome restores the 2x
#   recipes  v3e.yaml = the registry recipe (md5 checked) with its header comment lines removed; mnbt16k.yaml and
#            prof.yaml derive from it. RES and comment.md are checked against private.pat before the job ends.
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; comment.md + RESULT for the poster; 2x restored on exit.
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k74; K=$HOME/GEN-AI/backlog/k74
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/prefill-k74-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
REG=@qwen38-flashnext-1x/qwen3.8-flash-next-1x-dgx-spark
REGF=$HOME/.cache/sparkrun/registries/qwen38-flashnext-1x/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
V3E_MD5=c95d3939d0a394248f40ebd3af41f956
C=$HOME/GEN-AI/k31; CORPUS=$R/results/corpus-code.txt; ARMS="mnbt-01 mnbt-02"
PRIV=$(cat "$K/private.pat" 2>/dev/null)   # names that must not reach the results; kept out of RES
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

mkrecipes() { # registry v3e -> v3e.yaml (no header), mnbt16k.yaml, prof.yaml, arm dirs
  [ "$(md5sum < "$REGF" | cut -c1-32)" = $V3E_MD5 ] || { echo "registry recipe is not v3e ($V3E_MD5)"; return 1; }
  sed '/^#/d' "$REGF" > "$K/v3e.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-dgx-spark-k74-mnbt16k|" \
      -e "s|^  max_num_batched_tokens: 8192\$|  max_num_batched_tokens: 16384|" \
      -e "s|^env:\$|env:\n  XDG_CACHE_HOME: \"/cache/runtime/k74-mnbt16k\"\n  VLLM_CACHE_ROOT: \"/cache/runtime/k74-mnbt16k/vllm\"|" \
      "$K/v3e.yaml" > "$K/mnbt16k.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-dgx-spark-k74-prof|" \
      -e "s|^    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\\$|    --profiler-config '{{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/runtime/prof-k74\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}}' \\\\\n    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\|" \
      -e "s|^env:\$|env:\n  VLLM_CACHE_ROOT: \"/cache/runtime/vllm-k74-prof\"|" "$K/v3e.yaml" > "$K/prof.yaml"
  [ "$(diff "$K/v3e.yaml" "$K/mnbt16k.yaml" | grep -c '^>')" = 4 ] && grep -qx '  max_num_batched_tokens: 16384' "$K/mnbt16k.yaml" \
    || { echo "mnbt16k.yaml edits did not apply"; return 1; }
  [ "$(diff "$K/v3e.yaml" "$K/prof.yaml" | grep -c '^>')" = 3 ] && grep -q prof-k74 "$K/prof.yaml" || { echo "prof.yaml edits did not apply"; return 1; }
  local a; for a in $ARMS; do mkdir -p "$K/$a"; cp "$K/mnbt16k.yaml" "$K/$a/mnbt16k.yaml"
    { echo "# k74 arm on $([ $a = mnbt-01 ] && echo dgx-01 || echo dgx-02): v3e with max_num_batched_tokens 16384 (prefill chunk 2x), own cache dir, bake"
      echo "name: $a"; echo "recipe: mnbt16k.yaml"; echo "base: $K/v3e.yaml"; echo "bake: yes"; echo "hook: $K/hook.sh"; } > "$K/$a/thunderdome.arm"; done
  ! grep -rliE "$PRIV" "$K"/*.yaml "$K"/mnbt-0?/ "$K"/*.sh "$K"/*.py || { echo "private pattern in a job file"; return 1; }; }
preflight() {
  local ok=0
  mkrecipes || ok=1
  python3 "$K/cold.py" selftest > /dev/null || { echo "cold.py selftest failed"; ok=1; }
  [ -s "$CORPUS" ] && [ -x "$C/venv/bin/python" ] && [ -s "$C/lib/llm_decode_bench.py" ] && [ -e "$R/scripts/prof_summary.py" ] && [ -s "$TD" ] \
    || { echo "corpus, llm-inference-bench, prof_summary.py or thunderdome.sh missing"; ok=1; }
  command -v sparkrun > /dev/null || { echo "missing sparkrun"; ok=1; }
  rm -rf "$K/dry"; env RES="$K/dry" GATE=0 CHAIN_NO_RESTORE=1 bash "$TD" "$K/mnbt-01" "$K/mnbt-02" --dry-run > "$K/td-dry-run.txt" 2>&1
  grep -q REFUSED "$K/td-dry-run.txt" && { echo "thunderdome dry-run refused an arm ($K/td-dry-run.txt)"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  preflight; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  grep -E '^(dgx0|screen)' "$K/td-dry-run.txt" 2>/dev/null
  echo "plan: phase 1 ~25 min (dgx-01 v3e cold 16K/64K/128K x3 + standalone prefill; dgx-02 profile boot, 16K + 128K windows)"
  echo "      phase 2 thunderdome mnbt16k vs v3e on both Sparks ~115 min (bake ~14, 4 boots + hook 64K/128K x3);"
  echo "      + ~45 min split gate only on a PROMOTE; restore ~5 min. Total ~2 h 20 min, ~3 h 5 min with a gate"
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job"; cp "$K"/run.sh "$K"/cold.py "$K"/hook.sh "$K"/after_k73.sh "$K"/*.yaml "$RES/job/"

x() { if [ "$1" = $H1 ]; then shift; bash -c "$*"; else shift; ssh -n -o ConnectTimeout=10 $H2 "$*"; fi; }
cont() { x $1 "docker ps --format '{{.Names}}' | grep sparkrun | head -1"; }
cold() { timeout -k 30 ${4:-1800} python3 "$K/cold.py" run --base http://$1:8000 --corpus "$CORPUS" --lens $2 --reps 3 --seed $3 --out "$5"; }
up() { # host recipe dir: boot solo, wait for health, boot facts
  local h=$1 rec=$2 d=$3 s c; mkdir -p "$d"
  ( cd "$K" && timeout -k 30 900 sparkrun run "$rec" --hosts $h --solo --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s); until [ "$(health $h)" = 200 ]; do [ $(( $(date +%s) - s )) -gt 3600 ] && { echo "boot timeout" > "$d/FAILED"; return 1; }; sleep 15; done
  echo "boot $(( $(date +%s) - s )) s, recipe $rec" > "$d/boot.txt"; c=$(cont $h)
  x $h "docker exec $c cat /tmp/sparkrun_serve.log" > "$d/serve.log" 2>/dev/null
  grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' "$d/serve.log" >> "$d/boot.txt"
  grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$d/serve.log" >> "$d/boot.txt"
  x $h "docker inspect --format '{{.Config.Image}}' $c" >> "$d/boot.txt" 2>&1
  pong $h > "$d/pong.txt"; }
down() { timeout -k 30 300 sparkrun stop --all --hosts $1 >> "$2/sparkrun.log" 2>&1 9>&-; sleep 5
  x $1 "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" > /dev/null 2>&1; }
p1_measure() { # dgx-01: registry v3e, cold prefill then standalone prefill
  local d=$RES/p1-dgx01 kv; up $H1 "$REG" "$d" || return 1
  kv=$(grep -oE 'GPU KV cache size: [0-9,]+' "$d/boot.txt" | tr -dc 0-9)
  python3 "$K/cold.py" run --base http://$H1:8000 --corpus "$CORPUS" --lens 2048 --reps 1 --seed warm --out "$d/warm.json" > /dev/null 2>&1
  cold $H1 16384,65536,131072 p1 2400 "$d/cold.json" > "$d/cold.txt" 2>&1; echo "cold exit=$?" >> "$d/cold.txt"
  ( cd "$d" && LLM_BENCH_NO_UPDATE_CHECK=1 timeout --foreground -k 30 2400 "$C/venv/bin/python" "$C/lib/llm_decode_bench.py" \
      --host $H1 --port 8000 --model $MODEL --display-mode plain --no-resume --output "$d/lib-prefill.json" \
      --standalone-prefill --prefill-only --prefill-contexts 16k,64k,128k ${kv:+--kv-budget $kv} > "$d/lib-prefill.log" 2>&1 < /dev/null )
  echo "lib exit=$?" >> "$d/cold.txt"; down $H1 "$d"; }
p1_profile() { # dgx-02: profiler boot, a 16K window and a 128K window
  local d=$RES/p1-dgx02-prof c n; up $H2 "$K/prof.yaml" "$d" || return 1; c=$(cont $H2)
  python3 "$K/cold.py" run --base http://$H2:8000 --corpus "$CORPUS" --lens 2048 --reps 1 --seed warm --out "$d/warm.json" > /dev/null 2>&1
  for n in 16384 131072; do
    curl -s -m 60 -X POST http://$H2:8000/start_profile > "$d/start-$n.txt" 2>&1
    timeout -k 30 900 python3 "$K/cold.py" run --base http://$H2:8000 --corpus "$CORPUS" --lens $n --reps 1 --seed prof --out "$d/req-$n.json" > /dev/null 2>&1
    curl -s -m 1200 -X POST http://$H2:8000/stop_profile > "$d/stop-$n.txt" 2>&1; sleep 60; done
  x $H2 "docker exec $c sh -c 'cd /cache/runtime && tar -cz prof-k74 && rm -rf prof-k74'" > "$d/trace.tgz" 2>/dev/null
  ( cd "$d" && tar -xzf trace.tgz && rm -f trace.tgz ); down $H2 "$d"; }

st "running: phase 1 (measure)"
stop_all
p1_measure > "$RES/p1-dgx01.log" 2>&1 & p1=$!
p1_profile > "$RES/p1-dgx02.log" 2>&1 & p2=$!
CH=$p1; wait $p1; CH=$p2; wait $p2; CH=
stop_all

st "running: phase 2 (thunderdome mnbt16k vs v3e)"
child env RES="$RES" GPU_LOCK_HELD=1 timeout -k 600 32400 bash "$TD" "$K/mnbt-01" "$K/mnbt-02" > "$RES/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
V=$(head -1 "$RES/STATE" 2>/dev/null); log "thunderdome exit=$rc STATE: $V"

st "running: report"
pf() { python3 - "$@" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))["summary"] if sys.argv[1] != "-" else {}
try:
    l = json.load(open(sys.argv[2])).get("prefill") or {}
except (OSError, ValueError):
    l = {}
print("| prompt | cold TTFT mean s (3 fresh) | cold tok/s (prompt / mean TTFT) | each | standalone tok/s (llm-inference-bench) | samples |")
print("|---|---:|---:|---|---:|---:|")
for n in ("131072", "65536", "16384"):
    a = c.get(n) or {}; b = l.get(n) or {}
    print(f"| {int(n)//1024}K | {a.get('ttft_mean_s', 0):.2f} | {a.get('tok_s', 0):.0f} | {a.get('tok_s_each', '')} | {b.get('tok_per_sec', 0):.0f} | {b.get('samples', '')} |")
EOF
}
D1=$RES/p1-dgx01; D2=$RES/p1-dgx02-prof
{ echo "k74 $(TZ=Europe/Bucharest date '+%F %T %Z'): long-context prefill on the single Spark, v3e (registry recipe $REG)"
  echo; echo "== phase 1, dgx-01 solo v3e: both prefill definitions from one boot (cold test first, then llm-inference-bench)"
  echo "$(tr '\n' ' ' < "$D1/boot.txt" 2>/dev/null)"
  pf "$D1/cold.json" "$D1/lib-prefill.json" 2>&1
  echo "v3d reference (same bench, 2026-10-05): 2,182 / 2,066 / 1,880 tok/s at 16K / 64K / 128K"
  echo; echo "== phase 1, dgx-02 v3e + torch profiler: one cold request per window"
  echo "$(tr '\n' ' ' < "$D2/boot.txt" 2>/dev/null)"
  for n in 16384 131072; do echo "  $((n / 1024))K request: $(python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"][sys.argv[2]]; print("TTFT %.2f s, %.0f tok/s (profiler on)" % (s["ttft_mean_s"], s["tok_s"]))' "$D2/req-$n.json" $n 2>/dev/null || echo missing)"; done
  i=0; for t in $(find "$D2/prof-k74" -name '*.json*' 2>/dev/null | sort); do i=$((i + 1))
    echo; echo "-- window $i ($([ $i = 1 ] && echo 16K || echo 128K))"; python3 "$R/scripts/prof_summary.py" "k74-w$i" 1 "$t" 2>&1 | head -45; done
  echo; echo "== phase 2, thunderdome mnbt16k (max_num_batched_tokens 16384) vs v3e: exit $rc, $V"
  for a in $ARMS; do echo; echo "-- $a ($([ $a = mnbt-01 ] && echo dgx-01 || echo dgx-02))"
    sed -n '/^== acceptance/,$p' "$RES/$a/verdict.txt" 2>/dev/null | grep -vE '^(  self|  cross)'
    [ -f "$RES/$a/gate/summary.txt" ] && { echo "-- gate"; cat "$RES/$a/gate/summary.txt"; }; done
  echo; python3 "$K/cold.py" report "$RES"/mnbt-0? 2>&1; } > "$RES/k74.txt"
# nothing in the published dir may name the private source (poster publishes RES and comment.md)
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
H128=$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]["131072"]; print("%.0f" % s["tok_s"])' "$D1/cold.json" 2>/dev/null || echo "n/a")
S128=$(python3 -c 'import json,sys; print("%.0f" % json.load(open(sys.argv[1]))["prefill"]["131072"]["tok_per_sec"])' "$D1/lib-prefill.json" 2>/dev/null || echo "n/a")
VV=$(for a in $ARMS; do printf '%s=%s ' $a "$(tail -1 "$RES/$a/verdict.txt" 2>/dev/null | sed 's/.*VERDICT=//')"; done)
{ echo "k74 result: long-context prefill on the single Spark (v3e). Headline: 128K cold prefill $H128 tok/s (prompt tokens / mean TTFT over 3 fresh prompts), standalone prefill $S128 tok/s on the same boot."
  echo; echo "Phase 2 screened one config-only arm, \`max_num_batched_tokens\` 8192 -> 16384, against v3e on both Sparks: \`$VV\`. The hook lines at the end are the paired cold 64K/128K numbers per boot."
  echo; echo '```'; head -c 50000 "$RES/k74.txt"; echo '```'
  echo; echo "Raw data on dgx-01: \`$RES\`."; } > "$K/comment.md"
grep -qiE "$PRIV" "$K/comment.md" && { FINAL="FAILED: k74 comment.md matched the private pattern (not posted; $RES)"; mv "$K/comment.md" "$K/private/"; exit 1; }
[ -s "$D1/cold.json" ] || { FINAL="FAILED: k74 phase 1 cold test missing ($RES/p1-dgx01.log); thunderdome $V"; exit 1; }
FINAL="DONE: k74 128K cold $H128 tok/s, standalone $S128; mnbt16k $VV($RES/k74.txt)"
