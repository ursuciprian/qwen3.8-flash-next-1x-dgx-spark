#!/usr/bin/env bash
# k78-1x-prefill-mnbt16k-rerun (2026-10-09, 1x repo issue #8): rerun of k74 phase 2 (v3e with max_num_batched_tokens
# 8192 -> 16384 on the single Spark) plus the standalone prefill numbers k74 did not record. Queued last, after k75.
# Why k74 measured nothing (results/prefill-k74-20261009-1959):
#   phase 2  the arm's own cache dir (XDG_CACHE_HOME=/cache/runtime/k74-mnbt16k) started empty. The image seed fills
#            the b12x plan selections (the 289 v3e records of preparation/3205a613 arrived intact; 18 new 16K-shape plans
#            were measured) and the torch AOT, but not the compiled b12x kernels, which exist only in the host runtime cache.
#            So the arm bake compiled every b12x kernel it touched (~240 before the KV pool, ~180 on dgx-01 and ~350 on
#            dgx-02 after the 14 GiB pool was reserved; of the 420 kernels dgx-01 compiled, 135 exist in the v3e cache).
#            Each compile held tens of MB; MemAvailable fell below 2 GiB and the lib.sh memory guard removed the
#            container on both Sparks (dgx-01 20:19:43, 478 s into the boot; dgx-02 20:21:24, 543 s), so Thunderdome saw
#            a vanished container ("bake boot failed", no serve.log). Not a timeout (bake BOOT_MAX is 5400 s).
#   phase 1  standalone column 0 / n/a: MODEL is not set in lib.sh, so `set -u` ended the llm-inference-bench subshell
#            ("MODEL: unbound variable", lib exit=1). cold.py ran fine.
# Steps:
#   seed     (CPU) both Sparks: arm cache dir /cache/runtime/k78-mnbt16k = copy of the host's v3e b12x cache (plans +
#            compiled kernels, ~27 MB; saves ~1/3 of the compiles). torch AOT still comes from the image seed.
#   prebake  both Sparks in parallel: the arm recipe with a 6 GiB KV pool (prebake.yaml: same env, same cache dir), so
#            the remaining compiles for the 16K-token shapes run with ~8 GiB more headroom than k74 had. Up to 3 tries:
#            compiled kernels persist on disk, so a try killed by the guard leaves less for the next (the guard is
#            restarted after an abort). Serve log copied every 15 s (survives a guard kill); boot time, AOT lines,
#            b12x counts, min MemAvailable, pong per try.
#   measure  dgx-01 solo v3e (registry recipe): cold.py 16K/64K/128K x3, then llm-inference-bench --standalone-prefill
#            --prefill-only 16k,64k,128k (README definition) with --model set, both from one boot.
#   screen   Thunderdome mnbt16k vs v3e on each Spark whose prebake came up (bake: yes as in k74; the arm bake boot
#            should now find its kernels compiled), hook = cold 64K/128K x3 per boot, normal verdict, split gate on a PROMOTE (default).
#            Serve logs of every boot copied every 20 s to $RES/snap (serve-*.log: the poster does not publish them).
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; comment.md + RESULT for the poster; 2x restored on exit (lib.sh).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k78; K=$HOME/GEN-AI/backlog/$J
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k78-1x-prefill-mnbt16k-rerun-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
MODEL=qwen3.8-flash-next   # the k74 bug: lib.sh does not set it
REG=@qwen38-flashnext-1x/qwen3.8-flash-next-1x-dgx-spark
REGF=$HOME/.cache/sparkrun/registries/qwen38-flashnext-1x/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
V3E_BODY_MD5=607e17fbb6d48f54f081bb3462c7eba0   # v3e recipe without its # header lines (the header changed on 2026-10-09: release names)
# host dir behind /cache/runtime for the v3e recipe (same name on both Sparks, relative to $HOME) and its plan file
RC=.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-22c0994b
PLAN=b12x/compile/preparation/3205a6131cd09d438a33c655ab9993610bcdd9e05db54684a929b3f0a66153f8.json
ARMC=k78-mnbt16k; KV_PRE=6442450944   # 6 GiB prebake pool (~420K tokens at the v3e ratio; max_model_len 262144 fits)
C=$HOME/GEN-AI/k31; CORPUS=$R/results/corpus-code.txt; ARMS="mnbt-01 mnbt-02"
PRIV=$(cat "$K/private.pat" 2>/dev/null)   # names that must not reach the results; kept out of RES
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

x() { if [ "$1" = $H1 ]; then shift; bash -c "$*"; else shift; ssh -n -o ConnectTimeout=10 $H2 "$*"; fi; }
lab() { [ "$1" = $H1 ] && echo dgx01 || echo dgx02; }
mkrecipes() { # registry v3e -> v3e.yaml (no header), mnbt16k.yaml, prebake.yaml, arm dirs
  [ "$(sed '/^#/d' "$REGF" | md5sum | cut -c1-32)" = $V3E_BODY_MD5 ] || { echo "registry recipe body is not v3e ($V3E_BODY_MD5)"; return 1; }
  sed '/^#/d' "$REGF" > "$K/v3e.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-dgx-spark-k78-mnbt16k|" \
      -e "s|^  max_num_batched_tokens: 8192\$|  max_num_batched_tokens: 16384|" \
      -e "s|^env:\$|env:\n  XDG_CACHE_HOME: \"/cache/runtime/$ARMC\"\n  VLLM_CACHE_ROOT: \"/cache/runtime/$ARMC/vllm\"|" \
      "$K/v3e.yaml" > "$K/mnbt16k.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-dgx-spark-k78-prebake|" \
      -e "s|^  kv_cache_memory_bytes: 15032385536\$|  kv_cache_memory_bytes: $KV_PRE|" "$K/mnbt16k.yaml" > "$K/prebake.yaml"
  [ "$(diff "$K/v3e.yaml" "$K/mnbt16k.yaml" | grep -c '^>')" = 4 ] && grep -qx '  max_num_batched_tokens: 16384' "$K/mnbt16k.yaml" \
    || { echo "mnbt16k.yaml edits did not apply"; return 1; }
  [ "$(diff "$K/mnbt16k.yaml" "$K/prebake.yaml" | grep -c '^>')" = 2 ] && grep -qx "  kv_cache_memory_bytes: $KV_PRE" "$K/prebake.yaml" \
    || { echo "prebake.yaml edits did not apply"; return 1; }
  local a; for a in $ARMS; do mkdir -p "$K/$a"; cp "$K/mnbt16k.yaml" "$K/$a/mnbt16k.yaml"
    { echo "# k78 arm on $([ $a = mnbt-01 ] && echo dgx-01 || echo dgx-02): v3e with max_num_batched_tokens 16384, own cache dir seeded from v3e's b12x cache, prebaked, bake"
      echo "name: $a"; echo "recipe: mnbt16k.yaml"; echo "base: $K/v3e.yaml"; echo "bake: yes"; echo "hook: $K/hook.sh"; } > "$K/$a/thunderdome.arm"; done
  ! grep -rliE "$PRIV" "$K"/*.yaml "$K"/mnbt-0?/ "$K"/*.sh "$K"/*.py || { echo "private pattern in a job file"; return 1; }; }
preflight() {
  local ok=0 h
  mkrecipes || ok=1
  python3 "$K/cold.py" selftest > /dev/null || { echo "cold.py selftest failed"; ok=1; }
  [ -s "$CORPUS" ] && [ -x "$C/venv/bin/python" ] && [ -s "$C/lib/llm_decode_bench.py" ] && [ -s "$TD" ] \
    || { echo "corpus, llm-inference-bench or thunderdome.sh missing"; ok=1; }
  LLM_BENCH_NO_UPDATE_CHECK=1 timeout 120 "$C/venv/bin/python" "$C/lib/llm_decode_bench.py" --help 2>/dev/null | grep -q -- --prefill-only \
    || { echo "llm-inference-bench --help has no --prefill-only"; ok=1; }
  for h in $H1 $H2; do x $h "test -s \$HOME/$RC/$PLAN && test -d \$HOME/$RC/b12x/compile" \
    || { echo "$(lab $h): v3e b12x cache missing (~/$RC/$PLAN)"; ok=1; }; done
  command -v sparkrun > /dev/null || { echo "missing sparkrun"; ok=1; }
  rm -rf "$K/dry"; env RES="$K/dry" GATE=0 CHAIN_NO_RESTORE=1 bash "$TD" "$K/mnbt-01" "$K/mnbt-02" --dry-run > "$K/td-dry-run.txt" 2>&1
  grep -q REFUSED "$K/td-dry-run.txt" && { echo "thunderdome dry-run refused an arm ($K/td-dry-run.txt)"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  preflight; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  grep -E '^(dgx0|screen)' "$K/td-dry-run.txt" 2>/dev/null
  echo "plan: seed arm cache dirs (CPU, seconds); prebake arm with a 6 GiB KV pool on both Sparks ~10-15 min (up to 3 tries);"
  echo "      dgx-01 v3e cold 16K/64K/128K x3 + standalone prefill ~25 min;"
  echo "      thunderdome mnbt16k vs v3e on both Sparks ~115 min (bake ~10, 4 boots + hook 64K/128K x3);"
  echo "      + ~45 min split gate only on a PROMOTE; restore ~5 min. Total ~2 h 50 min, ~3 h 35 min with a gate"
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job"; cp "$K"/run.sh "$K"/cold.py "$K"/hook.sh "$K"/after_k75.sh "$K"/*.yaml "$RES/job/"

cid() { x $1 "docker ps -q --filter name=sparkrun | head -1"; }
snap() { # host outdir: copy the serve log of the running sparkrun container (by id: the last copy survives a guard kill)
  local c f; c=$(cid $1); [ -n "$c" ] || return 1; f=$2/serve-$(lab $1)-$c.log
  x $1 "docker exec $c cat /tmp/sparkrun_serve.log" > "$f.tmp" 2>/dev/null && mv -f "$f.tmp" "$f"; rm -f "$f.tmp"; }
snaploop() { mkdir -p "$RES/snap"; while kill -0 $$ 2>/dev/null; do snap $H1 "$RES/snap"; snap $H2 "$RES/snap"; sleep 20; done; }
seed() { # host: arm cache dir = copy of the v3e b12x cache (plans + compiled kernels)
  x $1 "set -e; B=\$HOME/$RC; rm -rf \"\$B/$ARMC\"; mkdir -p \"\$B/$ARMC\"; cp -a \"\$B/b12x\" \"\$B/$ARMC/\"
        test -s \"\$B/$ARMC/$PLAN\"; du -sh \"\$B/$ARMC\" | cut -f1; ls \"\$B/$ARMC/b12x/compile\" | wc -l"; }
srl() { flock -o "$RES/.sparkrun.lock" "$@"; }   # one sparkrun call at a time (as thunderdome.sh sr)
down() { srl timeout -k 30 300 sparkrun stop --all --hosts $1 >> "$2/sparkrun.log" 2>&1 9>&-; sleep 5
  x $1 "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" > /dev/null 2>&1; }
prebake() { # host try: arm recipe with the 6 GiB KV pool into the seeded arm cache dir, then stop
  local h=$1 d=$RES/prebake-$(lab $1)/try$2 s e c
  mkdir -p "$d"; cp "$K/prebake.yaml" "$d/recipe.yaml"
  ( cd "$K" && srl timeout -k 30 900 sparkrun run "$K/prebake.yaml" --hosts $h --solo --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  until [ "$(health $h)" = 200 ]; do
    c=$(cid $h); snap $h "$d"
    if [ $(( $(date +%s) - s )) -gt 120 ] && { [ -z "$c" ] || grep -qsE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' "$d"/serve-*.log; }; then
      echo "boot failed after $(( $(date +%s) - s )) s (container $([ -z "$c" ] && echo gone || echo up))" > "$d/FAILED"; down $h "$d"; return 1; fi
    [ $(( $(date +%s) - s )) -gt 5400 ] && { echo "boot timeout" > "$d/FAILED"; down $h "$d"; return 1; }
    sleep 15; done
  e=$(date +%s); snap $h "$d"
  { echo "boot $((e - s)) s, KV pool $KV_PRE bytes, min MemAvailable $(minmem "$RES/guard-$(lab $h).log" $s $e) GiB"
    grep -hm1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$d"/serve-*.log
    grep -hoE '(saved AOT|Directly load AOT|Dynamo bytecode transform time)[^ ]*' "$d"/serve-*.log | sort | uniq -c
    grep -hoE 'b12x ready [a-z_.]+: [0-9]+/[0-9]+ ready, [0-9]+ measured, [0-9]+ cached, [0-9]+ compilations' "$d"/serve-*.log | tail -4
    echo "pong: $(pong $h)"; } > "$d/boot.txt" 2>&1
  down $h "$d"; }
up() { # host recipe dir: boot solo, wait for health, boot facts
  local h=$1 rec=$2 d=$3 s c; mkdir -p "$d"
  ( cd "$K" && srl timeout -k 30 900 sparkrun run "$rec" --hosts $h --solo --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s); until [ "$(health $h)" = 200 ]; do [ $(( $(date +%s) - s )) -gt 3600 ] && { echo "boot timeout" > "$d/FAILED"; return 1; }; sleep 15; done
  echo "boot $(( $(date +%s) - s )) s, recipe $rec" > "$d/boot.txt"; c=$(cid $h)
  x $h "docker exec $c cat /tmp/sparkrun_serve.log" > "$d/serve.log" 2>/dev/null
  grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' "$d/serve.log" >> "$d/boot.txt"
  grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$d/serve.log" >> "$d/boot.txt"
  x $h "docker inspect --format '{{.Config.Image}}' $c" >> "$d/boot.txt" 2>&1
  pong $h > "$d/pong.txt"; }
p1_measure() { # dgx-01: registry v3e, cold prefill then standalone prefill
  local d=$RES/p1-dgx01 kv; up $H1 "$REG" "$d" || return 1
  kv=$(grep -oE 'GPU KV cache size: [0-9,]+' "$d/boot.txt" | tr -dc 0-9)
  python3 "$K/cold.py" run --base http://$H1:8000 --corpus "$CORPUS" --lens 2048 --reps 1 --seed warm --out "$d/warm.json" > /dev/null 2>&1
  timeout -k 30 2400 python3 "$K/cold.py" run --base http://$H1:8000 --corpus "$CORPUS" --lens 16384,65536,131072 --reps 3 --seed p1 \
    --out "$d/cold.json" > "$d/cold.txt" 2>&1; echo "cold exit=$?" >> "$d/cold.txt"
  ( cd "$d" && LLM_BENCH_NO_UPDATE_CHECK=1 timeout --foreground -k 30 2400 "$C/venv/bin/python" "$C/lib/llm_decode_bench.py" \
      --host $H1 --port 8000 --model $MODEL --display-mode plain --no-resume --output "$d/lib-prefill.json" \
      --standalone-prefill --prefill-only --prefill-contexts 16k,64k,128k ${kv:+--kv-budget $kv} > "$d/lib-prefill.log" 2>&1 < /dev/null )
  echo "lib exit=$?" >> "$d/cold.txt"; down $H1 "$d"; }

st "running: seed (arm cache dirs from the v3e b12x cache)"
stop_all
for h in $H1 $H2; do seed $h > "$RES/seed-$(lab $h).txt" 2>&1 || { FINAL="FAILED: seed on $(lab $h) ($RES/seed-$(lab $h).txt)"; exit 1; }; done

regard() { # restart a memory guard that aborted (it exits after removing the containers); the EXIT trap kills $G1 $G2
  kill -0 $G1 2>/dev/null || { bash -c "$GUARD" >> "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- & G1=$!; log "memory guard dgx-01 restarted"; }
  kill -0 $G2 2>/dev/null || { ssh -n $H2 "$GUARD" >> "$RES/guard-dgx02.log" 2>&1 9>&- & G2=$!; log "memory guard dgx-02 restarted"; }; }
r1=1; r2=1
for try in 1 2 3; do
  st "running: prebake try $try (arm, 6 GiB KV pool)"
  p1=; p2=; mkdir -p "$RES/prebake-dgx01" "$RES/prebake-dgx02"
  [ $r1 = 0 ] || { prebake $H1 $try > "$RES/prebake-dgx01/try$try.log" 2>&1 & p1=$!; }
  [ $r2 = 0 ] || { prebake $H2 $try > "$RES/prebake-dgx02/try$try.log" 2>&1 & p2=$!; }
  [ -n "$p1" ] && { CH=$p1; wait $p1; r1=$?; }
  [ -n "$p2" ] && { CH=$p2; wait $p2; r2=$?; }
  CH=; log "prebake try $try: dgx01 rc=$r1 dgx02 rc=$r2"
  [ $r1 = 0 ] && [ $r2 = 0 ] && break
  regard; done
regard
OK=; [ $r1 = 0 ] && OK="$K/mnbt-01"; [ $r2 = 0 ] && OK="$OK $K/mnbt-02"

st "running: measure (dgx-01 v3e, cold + standalone prefill)"
child p1_measure > "$RES/p1-dgx01.log" 2>&1
stop_all

rc=-; V="not run: no prebake came up"
if [ -n "$OK" ]; then
  st "running: screen (thunderdome mnbt16k vs v3e:$(for a in $OK; do printf ' %s' "$(basename "$a")"; done))"
  snaploop & SN=$!
  child env RES="$RES" GPU_LOCK_HELD=1 timeout -k 600 32400 bash "$TD" $OK > "$RES/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
  kill $SN 2>/dev/null; rm -f "$RES"/snap/*.tmp
  V=$(head -1 "$RES/STATE" 2>/dev/null); log "thunderdome exit=$rc STATE: $V"; fi

st "running: report"
pf() { python3 - "$@" <<'EOF'
import json, sys
try:
    c = json.load(open(sys.argv[1]))["summary"]
except (OSError, ValueError, KeyError):
    c = {}
try:
    l = json.load(open(sys.argv[2])).get("prefill") or {}
except (OSError, ValueError):
    l = {}
print("| prompt | cold TTFT mean s (3 fresh) | cold tok/s (prompt / mean TTFT) | each | standalone tok/s (llm-inference-bench) | server-side tok/s | samples |")
print("|---|---:|---:|---|---:|---:|---:|")
for n in ("131072", "65536", "16384"):
    a = c.get(n) or {}; b = l.get(n) or {}
    print(f"| {int(n)//1024}K | {a.get('ttft_mean_s', 0):.2f} | {a.get('tok_s', 0):.0f} | {a.get('tok_s_each', '')} | "
          f"{b.get('tok_per_sec', 0):.0f} | {(b.get('server_validation') or {}).get('tok_per_sec', 0):.0f} | {b.get('samples', '')} |")
EOF
}
D1=$RES/p1-dgx01
{ echo "k78 $(TZ=Europe/Bucharest date '+%F %T %Z'): rerun of k74 phase 2 (v3e, max_num_batched_tokens 16384, single Spark) + standalone prefill"
  echo; echo "== seed: arm cache dir /cache/runtime/$ARMC = copy of the v3e b12x cache (size, kernel dirs)"
  for h in dgx01 dgx02; do echo "$h: $(tr '\n' ' ' < "$RES/seed-$h.txt")"; done
  echo; echo "== prebake: arm recipe with a 6 GiB KV pool"
  for h in dgx01 dgx02; do for t in "$RES/prebake-$h"/try?; do [ -d "$t" ] || continue
    echo "-- $h $(basename "$t") $(cat "$t/FAILED" 2>/dev/null)"; cat "$t/boot.txt" 2>/dev/null; done; done
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'
  echo; echo "== dgx-01 solo v3e (registry recipe $REG): both prefill definitions from one boot (cold test first, then llm-inference-bench)"
  echo "$(tr '\n' ' ' < "$D1/boot.txt" 2>/dev/null)"
  pf "$D1/cold.json" "$D1/lib-prefill.json" 2>&1
  echo "k74 cold test on v3e (2026-10-09): 2,049 / 1,966 / 1,797 tok/s at 16K / 64K / 128K"
  echo "v3d reference (llm-inference-bench standalone, 2026-10-05): 2,182 / 2,066 / 1,880 tok/s at 16K / 64K / 128K"
  echo; echo "== thunderdome mnbt16k (max_num_batched_tokens 16384) vs v3e: exit $rc, $V"
  for a in $ARMS; do echo; echo "-- $a ($([ $a = mnbt-01 ] && echo dgx-01 || echo dgx-02))"
    cat "$RES/$a/bake-arm/boot.txt" 2>/dev/null | sed 's/^/bake-arm: /'
    sed -n '/^== acceptance/,$p' "$RES/$a/verdict.txt" 2>/dev/null | grep -vE '^(  self|  cross)'
    [ -f "$RES/$a/gate/summary.txt" ] && { echo "-- gate"; cat "$RES/$a/gate/summary.txt"; }; done
  echo; python3 "$K/cold.py" report "$RES"/mnbt-0? 2>&1; } > "$RES/k78.txt"
# nothing in the published dir may name the private source (poster publishes RES and comment.md)
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
sv() { python3 -c 'import json,sys; print("%.0f" % json.load(open(sys.argv[1]))["prefill"][sys.argv[2]]["tok_per_sec"])' "$D1/lib-prefill.json" $1 2>/dev/null || echo "n/a"; }
cv() { python3 -c 'import json,sys; print("%.0f" % json.load(open(sys.argv[1]))["summary"][sys.argv[2]]["tok_s"])' "$D1/cold.json" $1 2>/dev/null || echo "n/a"; }
S16=$(sv 16384); S64=$(sv 65536); S128=$(sv 131072); H128=$(cv 131072)
VV=$(for a in $ARMS; do printf '%s=%s ' $a "$(tail -1 "$RES/$a/verdict.txt" 2>/dev/null | sed 's/.*VERDICT=//')"; done)
{ echo "k78 result: rerun of the k74 \`max_num_batched_tokens\` 8192 -> 16384 arm on the single Spark (v3e), plus the standalone prefill numbers k74 did not record."
  echo; echo "Standalone prefill (llm-inference-bench, the README definition): $S16 / $S64 / $S128 tok/s at 16K / 64K / 128K. Cold prefill on the same boot (prompt tokens / mean TTFT over 3 fresh prompts): 128K $H128 tok/s."
  echo; echo "This time the arm's cache dir started as a copy of the v3e b12x cache and the arm was prebaked with a 6 GiB KV pool before the screen. Screen verdicts: \`$VV\`. The hook lines at the end are the paired cold 64K/128K numbers per boot."
  echo; echo '```'; head -c 50000 "$RES/k78.txt"; echo '```'
  echo; echo "Raw data on dgx-01: \`$RES\`."; } > "$K/comment.md"
grep -qiE "$PRIV" "$K/comment.md" && { FINAL="FAILED: k78 comment.md matched the private pattern (not posted; $RES)"; mv "$K/comment.md" "$K/private/"; exit 1; }
[ "$S128" = n/a ] && S128="n/a (standalone failed, $D1/lib-prefill.log)"
FINAL="DONE: k78 standalone 128K $S128 tok/s, cold 128K $H128; prebake dgx01 rc=$r1 dgx02 rc=$r2; mnbt16k ${VV:-not screened }($RES/k78.txt)"
