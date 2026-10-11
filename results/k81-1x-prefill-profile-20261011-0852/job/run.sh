#!/usr/bin/env bash
# k81-1x-prefill-profile (2026-10-11, 1x #27): where does a cold 16K prefill spend ~490 us per prompt token on the
# single Spark? torch-profiler trace of one cold 16K prompt (c1, max_num_batched_tokens 8192 as shipped: 2 full chunks
# + a short tail) per arm, op-class breakdown per chunk (k81an.py), plus 3 unprofiled cold 16K TTFTs per boot.
#   dgx-01   1x v2.2.0 from the sparkrun registry (header stripped), each boot solo:
#              ship   as shipped (14 GiB KV pool, real runtime cache; the profiler config is not a compile factor, so
#                     the warm AOT loads) + --profiler-config
#              min0   VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS=0: GDN projections on the W4A16 NVFP4 path at prefill too
#              min1   MIN_TOKENS=1: MXFP8 at every M (prefill path = 41 by construction; a control)
#              nomtp  no --speculative-config, no VLLM_GDN_DEFERRED_CHECKPOINTS (k85: it refuses to start without MTP)
#            the three arms: 6 GiB KV pool (k74/k78 OOM lesson: compiles after the pool is reserved) and their own
#            runtime cache /cache/runtime/k81-a (b12x dir seeded from the v2.2.0 cache; MIN_TOKENS is a compile factor)
#   dgx-02   (in parallel) b12x rebase image ghcr.io/ursuciprian/spark-vllm-b12x:rb-20261010-254212b1-12aef63d (vLLM
#            12aef63d = the 1x vLLM 5dad364d + 2 commits), own empty cache /cache/runtime/k81-rb (new b12x selection
#            schema), 6 GiB pool: prebake boot (cold autotune) up to 3 tries, then the measured boot (must measure 0
#            plans); then the shipped image with the same 6 GiB pool and a seeded cache (ship6), so the rebase delta is
#            same Spark, same pool. Rebase MIN_TOKENS=0 / MTP-off arms skipped to keep the 2x downtime short.
#   per boot warm-up (2K + one 16K), TTFT 16K x3 (cold.py: nonce + own corpus window per request, max_tokens 1),
#            /start_profile, one more cold 16K, /stop_profile, trace tarred out of the container. Each unit reads its own
#            corpus char range (no prompt repeats, no page-cache reuse of a prompt across boots).
# ETA ~75 min: stop 2x 1, seed 1; dgx-01 ship ~6, min0/min1 ~12 each (torch compile + a few plans), nomtp ~15;
#   dgx-02 prebake ~15-30 (k83 TP=2 prebake 839 s), rb ~6, ship6 ~8; report 1; restore 2x ~6.
# Output: $RES/d0N-<arm>/ (cold-*.json, trace dir prof-k81-<arm>/ kept on dgx-01, the poster excludes prof-k81-*),
# k81.txt, comment.md here + RESULT. Notes for the comment: $K/notes.md (optional, read at report time).
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k81; K=$HOME/GEN-AI/backlog/k81
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k81-1x-prefill-profile-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, boot, serve_log, MODEL
REG1=qwen38-flashnext-1x
REGF=$HOME/.cache/sparkrun/registries/$REG1/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
SHIP_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-d3-hf-20261010-21e0b201-5dad364d-warm
RB_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:rb-20261010-254212b1-12aef63d
# host dir behind /cache/runtime for ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE@03f4a057 (sparkrun model_key)
RC=.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923
KV_ARM=6442450944; CORPUS=$R/results/corpus-code.txt; RD=$K/rec
PRIV=$(cat "$K/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

mkrec() { # label [env edits as KEY=VALUE | -KEY] [kv] [nomtp] [rb] -> $RD/<label>.yaml
  local l=$1 o=$RD/$1.yaml c=$2 kv=$3 mtp=$4 img=$5 prof
  prof="    --profiler-config '{{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/runtime/prof-k81-$l\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}}' \\\\"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k81-$l|" \
      -e "s|^    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\\$|$prof\n    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\|" "$RD/base.yaml" > "$o"
  [ -n "$c" ] && sed -i "s|^env:\$|env:\n  XDG_CACHE_HOME: \"/cache/runtime/$c\"\n  VLLM_CACHE_ROOT: \"/cache/runtime/$c/vllm\"|" "$o"
  [ -n "$kv" ] && sed -i "s|^  kv_cache_memory_bytes: 15032385536\$|  kv_cache_memory_bytes: $kv|" "$o"
  [ "$mtp" = nomtp ] && sed -i -e '/^    --speculative-config /d' -e '/^  VLLM_GDN_DEFERRED_CHECKPOINTS: /d' "$o"
  [ -n "$img" ] && sed -i "s|^container: .*|container: $img|" "$o"
  echo "$o"; }
mkrecipes() {
  local ok=0 f
  mkdir -p "$RD"
  [ "$(head -1 "$REGF" 2>/dev/null)" = "# Release: v2.2.0" ] || { echo "registry 1x recipe is not v2.2.0 ($REGF)"; return 1; }
  sed '/^#/d' "$REGF" > "$RD/base.yaml"
  [ "$(sed -n 's/^container: *//p' "$RD/base.yaml")" = "$SHIP_IMG" ] || { echo "registry 1x container is not $SHIP_IMG"; return 1; }
  mkrec ship "" "" mtp "" > /dev/null
  mkrec min0 k81-a $KV_ARM mtp "" > /dev/null; sed -i 's|^  VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "41"$|  VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "0"|' "$RD/min0.yaml"
  mkrec min1 k81-a $KV_ARM mtp "" > /dev/null; sed -i 's|^  VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "41"$|  VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "1"|' "$RD/min1.yaml"
  mkrec nomtp k81-a $KV_ARM nomtp "" > /dev/null
  mkrec rb k81-rb $KV_ARM mtp "$RB_IMG" > /dev/null
  mkrec ship6 k81-a $KV_ARM mtp "" > /dev/null
  # expected diff vs base (added, removed): name +1-1, profiler +1, cache env +2, kv +1-1, MIN or image +1-1, nomtp -2
  chk() { local a d; a=$(diff "$RD/base.yaml" "$RD/$1.yaml" | grep -c '^>'); d=$(diff "$RD/base.yaml" "$RD/$1.yaml" | grep -c '^<')
    [ "$a $d" = "$2" ] && echo "ok   $1: +$a -$d" || { echo "BAD  $1: +$a -$d (want $2)"; ok=1; }; }
  chk ship "2 1"; chk min0 "6 3"; chk min1 "6 3"; chk nomtp "5 4"; chk rb "6 3"; chk ship6 "5 2"
  grep -q 'VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "0"' "$RD/min0.yaml" && grep -q 'VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: "1"' "$RD/min1.yaml" \
    && ! grep -q 'speculative-config\|GDN_DEFERRED' "$RD/nomtp.yaml" && grep -q "^container: $RB_IMG\$" "$RD/rb.yaml" \
    && grep -q "prof-k81-ship" "$RD/ship.yaml" || { echo "recipe edits did not apply"; ok=1; }
  for f in "$RD"/*.yaml; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }
preflight() {
  local ok=0 h
  mkrecipes || ok=1
  python3 "$K/cold.py" selftest > /dev/null || { echo "cold.py selftest failed"; ok=1; }
  python3 "$K/k81an.py" selftest > /dev/null || { echo "k81an.py selftest failed"; ok=1; }
  [ -s "$CORPUS" ] || { echo "corpus missing"; ok=1; }
  command -v sparkrun > /dev/null || { echo "missing sparkrun"; ok=1; }
  for h in $H1 $H2; do
    x $h "test -d \$HOME/$RC/b12x/compile/preparation" || { echo "$(hn $h): v2.2.0 runtime cache ~/$RC missing"; ok=1; }
    x $h "docker image inspect $SHIP_IMG > /dev/null 2>&1" || { echo "$(hn $h): $SHIP_IMG missing"; ok=1; }; done
  x $H2 "docker image inspect $RB_IMG > /dev/null 2>&1" || { echo "dgx02: $RB_IMG missing"; ok=1; }
  python3 -c "import hashlib; print('ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-' + hashlib.sha256(b'ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE@03f4a0570496bbe741189f4a2a7d611679e85406').hexdigest()[:8])" \
    | grep -qx "${RC##*/}" || { echo "runtime cache key mismatch"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  timeout 300 sparkrun registry update $REG1 > /dev/null 2>&1 < /dev/null || echo "note: registry update failed (run retries)"
  preflight; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: stop 2x; dgx-01 ship (14 GiB, warm) then min0, min1, nomtp (6 GiB, k81-a cache); dgx-02 rebase prebake (<=3 tries) + rb, then ship6"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: preflight"
timeout 300 sparkrun registry update $REG1 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job"; cp "$K/run.sh" "$K/cold.py" "$K/k81an.py" "$RD"/*.yaml "$RES/job/"

regard() { # host: restart that Spark's memory guard if it aborted (it exits after removing the containers). Each Spark's
  # sequence restarts only its own guard; a restarted guard is capped at 3 h (the EXIT trap only knows $G1 $G2).
  local h=$1 pf=$RES/.guard-$(hn $1).pid p
  p=$(cat "$pf" 2>/dev/null); [ -n "$p" ] || p=$([ $h = $H1 ] && echo $G1 || echo $G2)
  kill -0 "$p" 2>/dev/null && return 0
  if [ $h = $H1 ]; then timeout 3h bash -c "$GUARD" >> "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- &
  else timeout 3h ssh -n $H2 "$GUARD" >> "$RES/guard-dgx02.log" 2>&1 9>&- & fi
  echo $! > "$pf"; log "memory guard $(hn $h) restarted"; }
down() { # host
  flock "$RES/.sparkrun.lock" timeout -k 30 300 sparkrun stop --all --hosts $1 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-; sleep 5
  x $1 "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" > /dev/null 2>&1; }
facts() { # unit host -> $RES/<unit>/facts.txt
  local d=$RES/$1 f; f=$(ls "$d"/serve-*.log 2>/dev/null | head -1)
  { printf '%s ' "$(sed -n 's/^container: *//p' "$d/recipe.yaml" | sed 's|.*:||')"
    printf 'MIN_TOKENS %s, ' "$(sed -n 's/^  VLLM_B12X_NVFP4_MXFP8_MIN_TOKENS: *//p' "$d/recipe.yaml" | tr -d '"')"
    grep -q 'speculative-config' "$d/recipe.yaml" && printf 'MTP 4, ' || printf 'MTP off, '
    printf 'KV %s GiB, ' "$(( $(sed -n 's/^  kv_cache_memory_bytes: *//p' "$d/recipe.yaml") / 1073741824 ))"
    printf '%s, ' "$(sed -n 's/^boot_s /boot /p' "$d/boot.txt" 2>/dev/null) s"
    [ -n "$f" ] && printf '%s, %s plans measured' "$(grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' "$f" | sed 's/Directly load AOT compilation/AOT loaded/; s/Dynamo bytecode transform time/torch compiled/')" \
      "$(grep -ohE '\b[0-9]+ measured' "$f" | awk '{s+=$1} END {print s+0}')"; } > "$d/facts.txt" 2>/dev/null; }
measure() { # unit host cskip
  local u=$1 h=$2 o=$3 d=$RES/$1 c
  cr() { timeout -k 30 900 python3 "$K/cold.py" run --base http://$h:8000 --corpus "$CORPUS" --lens "$1" --reps "$2" --seed "$u-$3" --cskip $(( o + $4 )) --out "$d/cold-$3.json"; }
  cr 2048 1 warm 0 > /dev/null 2>&1; cr 16384 1 w16 10000 > /dev/null 2>&1
  cr 16384 3 ttft 80000 > "$d/cold-ttft.txt" 2>&1
  curl -s -m 60 -X POST http://$h:8000/start_profile > "$d/start_profile.txt" 2>&1
  cr 16384 1 prof 300000 > "$d/cold-prof.txt" 2>&1
  curl -s -m 900 -X POST http://$h:8000/stop_profile > "$d/stop_profile.txt" 2>&1; sleep 20
  c=$(x $h "docker ps --format '{{.Names}}' | grep sparkrun | head -1")
  x $h "docker exec $c sh -c 'cd /cache/runtime && tar -cz prof-k81-${u#d0?-} && rm -rf prof-k81-${u#d0?-}'" > "$d/trace.tgz" 2>/dev/null
  ( cd "$d" && tar -xzf trace.tgz && rm -f trace.tgz )
  serve_log $h "$d/serve-$(hn $h).log"; facts "$u" "$h"
  log "$u: $(tail -1 "$d/cold-ttft.txt" | cut -c1-120); trace files: $(find "$d" -name '*.json*' -path '*prof-k81*' | wc -l)"; }
arm() { # unit host recipe cap cskip
  local u=$1 h=$2
  if boot "$u" "$3" "$4" "$h"; then measure "$u" "$h" "$5"; else echo "boot failed" > "$RES/$u/FAILED"; facts "$u" "$h"; log "$u: boot failed"; fi
  down "$h"; regard "$h"; }
seq01() {
  arm d01-ship $H1 "$RD/ship.yaml" 1800 0
  arm d01-min0 $H1 "$RD/min0.yaml" 3600 400000
  arm d01-min1 $H1 "$RD/min1.yaml" 3600 800000
  arm d01-nomtp $H1 "$RD/nomtp.yaml" 4500 1200000; }
seq02() {
  local t ok=
  for t in 1 2 3; do
    if boot d02-rb-prebake-try$t "$RD/rb.yaml" 5400 $H2; then facts d02-rb-prebake-try$t $H2; down $H2; ok=1; break; fi
    facts d02-rb-prebake-try$t $H2; down $H2; regard $H2; log "rb prebake try $t failed"; done
  [ -n "$ok" ] && arm d02-rb $H2 "$RD/rb.yaml" 1800 1600000 || log "rb: 3 prebake tries failed, not measured"
  arm d02-ship6 $H2 "$RD/ship6.yaml" 3600 2000000; }

st "running: stop 2x, seed caches"
stop_all
for h in $H1 $H2; do
  x $h "set -e; B=\$HOME/$RC; rm -rf \"\$B/k81-a\" \"\$B/k81-rb\"; mkdir -p \"\$B/k81-a\" \"\$B/k81-rb\"; cp -a \"\$B/b12x\" \"\$B/k81-a/\"; du -sh \"\$B/k81-a\" | cut -f1" \
    > "$RES/seed-$(hn $h).txt" 2>&1 || { FINAL="FAILED: seed on $(hn $h) ($RES/seed-$(hn $h).txt)"; exit 1; }; done
st "running: dgx-01 ship/min0/min1/nomtp, dgx-02 rebase prebake + rb + ship6"
seq01 > "$RES/seq-dgx01.log" 2>&1 & p1=$!
seq02 > "$RES/seq-dgx02.log" 2>&1 & p2=$!
CH="$p1 $p2"; wait $p1; wait $p2; CH=
for f in "$RES"/.guard-*.pid; do [ -e "$f" ] && kill "$(cat "$f")" 2>/dev/null; done

st "running: report"
python3 "$K/k81an.py" report "$RES" > "$RES/k81.txt" 2>&1
{ echo; echo "boot facts per unit:"; for u in "$RES"/d0?-*/; do echo "  $(basename "$u"): $(cat "$u/facts.txt" 2>/dev/null) $(cat "$u/FAILED" 2>/dev/null)"; done
  for l in dgx01 dgx02; do echo "guard $l: min MemAvailable $(awk '$2 ~ /^[0-9]+$/' "$RES/guard-$l.log" | sort -k2 -n | head -1 | awk '{printf "%.2f GiB", $2/1024}')"; done
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } >> "$RES/k81.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
n=$(find "$RES" -path '*prof-k81-*' -name '*.json*' | wc -l)
t16=$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]["16384"]; print("%.2f s, %.0f us/token" % (s["ttft_mean_s"], 1e6 / s["tok_s"]))' "$RES/d01-ship/cold-ttft.json" 2>/dev/null || echo n/a)
{ echo "k81 result: torch-profiler breakdown of one cold 16K prefill on 1x v2.2.0 (c1, max_num_batched_tokens 8192, so 2 full chunks and a short tail), with MIN_TOKENS 0/1 and MTP-off arms on dgx-01 and the b12x rebase image against the shipped image on dgx-02 (same Spark, same 6 GiB KV pool)."
  [ -s "$K/notes.md" ] && { echo; cat "$K/notes.md"; }
  echo; echo '```'; head -c 55000 "$RES/k81.txt"; echo '```'
  echo; echo "Traces stay on dgx-01 in \`$RES\` (prof-k81-*). Results: {RESULTS_URL}"; } > "$K/comment.md"
grep -qiE "$PRIV" "$K/comment.md" && { mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k81 comment.md matched the private pattern (not posted; $RES)"; exit 1; }
cp "$K/comment.md" "$RES/comment.md"
[ "$n" -ge 1 ] || { FINAL="FAILED: k81 no traces ($RES/k81.txt)"; exit 1; }
FINAL="DONE: k81 $n traces, ship TTFT 16K $t16 ($RES/k81.txt)"
exit 0
