#!/usr/bin/env bash
# k90-tile-fix (2026-10-11, 1x #27): b12x large-row MXFP8 capacity regime vs the shipped 1x v2.2.0.
# Fix: b12x fix/pf-large-m-regime b45fe867 (base 21e0b201, the shipped b12x). A packed MXFP8 declaration of >= 4096 rows
# with K >= 4096 and N > 1536 on low-SM parts (vLLM's max_num_batched_tokens capacity regime of GDN out_proj and
# attention o_proj, K6144 N2560) lowers to 128x128/BK64 without the large-M unroll, and plan_regimes adds a 4095-row
# capacity regime with the previous plan (64x128/BK128) that serves every live M below 4096. Decode (exact-M) and
# M < 4096 run the shipped kernels. k88 P0 (dense GEMM alone): 1.77 vs 3.64 ms at M 6048, 2.36 vs 4.88 at 8192.
# Image: $K/IMAGE (tf-20261011-b45fe867-5dad364d, hosted CI build-b12x-overlay on the 1x v2.2.0 image; torch AOT seed
# removed by the overlay, so the arm compiles once in its own cache during the prebake).
#   phase 0  (dgx-01, serving stopped) gemm_probe.py through blockscaled.plan_regimes as vLLM declares it, in the
#            shipped image and in the arm image: ms per live M (1264..8192), the regime plan per M, output sha1 per M
#            (M < 4096 must be byte-identical across images), K6144 N2560 cos/max-rel at M >= 4536, other dense shapes
#            of the model (TP=1 and the TP=2 out_proj width) for the same cliff. CHECK=FAIL stops the job.
#   1x       control = shipped 1x v2.2.0 (registry, header stripped) + --profiler-config; arm = the same on the arm image
#            with its own runtime cache k90-tf (b12x dir seeded from the v2.2.0 cache, prebaked with a 6 GiB KV pool on
#            both Sparks; measured boots must show 0 plans measured). Job-local Thunderdome on both Sparks, ctl arm arm
#            ctl: T=0 probes fresh c1/c4/c8, llama-benchy pp2048 c1 + tg512 c1/c8 (3 runs), hook: cold TTFT
#            16K/64K/128K x3, boot facts, pass-1 torch-profiler window (out_proj/o_proj ms per 6048 chunk).
#   2x       only when phase 0 shows a TP=2 projection whose plan changes (K3072 N2560 is outside K >= 4096, so the 2x
#            arm would run the shipped plans); otherwise skipped with that reason in k90.txt.
#   judge    per Spark: pool.py (RULE=PASS|FLAT|FAIL; no cell worse than its noise band) + cold TTFT vs the control's
#            boot-to-boot spread (cold_judge.py).
# ETA ~2 h (pull 5, phase 0 ~15, prebake ~20, screen ~80) + restore ~6 min.
# Output: $RES/{p0,prebake-*,tf-01,tf-02,pooled-*,...}, k90.txt; comment.md here + RESULT. Nothing ships from this job.
# Contract (backlog/runner.sh): caller holds both GPU locks (no NEEDS line: both Sparks); exit 75 untouched while another
# job is queued/running; STATE first line DONE/FAILED; EXIT trap restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | through the runner (JOBS=k90). Stop: kill -TERM <pid>.
set -u
J=k90; K=$HOME/GEN-AI/backlog/k90
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k90-1x-tile-fix-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, MODEL
REG1=qwen38-flashnext-1x
REGF=$HOME/.cache/sparkrun/registries/$REG1/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
SHIP_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-d3-hf-20261010-21e0b201-5dad364d-warm
PF_IMG=$(cat "$K/IMAGE" 2>/dev/null)   # tile-fix arm image
RC=.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923   # 1x model_key (k81)
KV_PRE=6442450944
CORPUS=$R/results/corpus-code.txt; RD=$K/rec
PRIV=$(cat "$K/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

mkrecs() { # base.yaml (+profiler), tf.yaml (arm image, cache k90-tf), its prebake; arm dirs
  mkdir -p "$RD"
  [ "$(head -1 "$REGF" 2>/dev/null)" = "# Release: v2.2.0" ] || { echo "registry 1x recipe is not v2.2.0 ($REGF)"; return 1; }
  [ "$(sed -n 's/^container: *//p' "$REGF")" = "$SHIP_IMG" ] || { echo "registry 1x container is not $SHIP_IMG"; return 1; }
  [ -n "$PF_IMG" ] || { echo "no arm image in $K/IMAGE"; return 1; }
  sed '/^#/d' "$REGF" | sed "s|^    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\\$|    --profiler-config '{{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/runtime/prof-k90\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}}' \\\\\n    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\|" > "$RD/base.yaml"
  arm() { # name cache mnbt
    sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k90-$1|" -e "s|^container: .*|container: $PF_IMG|" \
        -e "s|^  max_num_batched_tokens: 8192\$|  max_num_batched_tokens: $3|" \
        -e "s|^env:\$|env:\n  XDG_CACHE_HOME: \"/cache/runtime/$2\"\n  VLLM_CACHE_ROOT: \"/cache/runtime/$2/vllm\"|" "$RD/base.yaml" > "$RD/$1.yaml"
    sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k90-$1-prebake|" \
        -e "s|^  kv_cache_memory_bytes: 15032385536\$|  kv_cache_memory_bytes: $KV_PRE|" "$RD/$1.yaml" > "$RD/$1-prebake.yaml"; }
  arm tf k90-tf 8192
  [ "$(sed '/^#/d' "$REGF" | diff - "$RD/base.yaml" | grep -c '^>')" = 1 ] || { echo "base.yaml profiler edit did not apply"; return 1; }
  [ "$(diff "$RD/base.yaml" "$RD/tf.yaml" | grep -c '^>')" = 4 ] && grep -qx "container: $PF_IMG" "$RD/tf.yaml" || { echo "tf.yaml edits did not apply"; return 1; }
  for a in tf; do [ "$(diff "$RD/$a.yaml" "$RD/$a-prebake.yaml" | grep -c '^>')" = 2 ] || { echo "$a-prebake.yaml edits did not apply"; return 1; }; done
  local s a n; for s in tf; do for n in 01 02; do a=$K/$s-$n; mkdir -p "$a"
    { echo "# k90 arm on dgx-$n: 1x v2.2.0 on the b12x large-row tile-fix image, own cache, prebaked; both sides with --profiler-config"
      echo "name: $s-$n"; echo "recipe: $RD/$s.yaml"; echo "base: $RD/base.yaml"; echo "bake: yes"; echo "hook: $K/hook.sh"; } > "$a/thunderdome.arm"; done; done; }
mktd() { # job-local thunderdome (as k88): fixed repo root, probe cells, benchy runs 3
  sed -e '/^R=\$(cd "\$(dirname /c\R='"$R" \
      -e '/^CELLS="fresh-c1/,/count-c8:0:8:2:0:300:20"$/c\CELLS="fresh-c1:256:1:2:104729:300:45 fresh-c4:256:4:2:314187:400:85 fresh-c8:256:8:2:418916:500:120"' \
      -e 's|--concurrency 1 8 --runs 2 |--concurrency 1 8 --runs 3 |' "$TD" > "$K/td.sh"
  [ "$(diff "$TD" "$K/td.sh" | grep -c '^>')" = 3 ] && grep -qx "R=$R" "$K/td.sh" && grep -q -- '--runs 3 ' "$K/td.sh" && bash -n "$K/td.sh" \
    || { echo "td.sh edits did not apply"; return 1; }; }
preflight() { # $1 = dry: the pf image may not be pulled yet
  local ok=0 h f
  for f in "$TD" "$R/scripts/thunderdome_report.py" "$K/pool.py" "$K/cold.py" "$K/cold_judge.py" "$K/gemm_probe.py" "$K/hook.sh" "$BL/k81/k81an.py" "$CORPUS"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  mkrecs || ok=1
  mktd || ok=1
  python3 "$K/pool.py" --selftest > /dev/null || { echo "pool.py selftest failed"; ok=1; }
  python3 "$K/cold.py" selftest > /dev/null || { echo "cold.py selftest failed"; ok=1; }
  python3 "$K/cold_judge.py" --selftest > /dev/null || { echo "cold_judge.py selftest failed"; ok=1; }
  bash -n "$K/hook.sh" || { echo "hook.sh syntax"; ok=1; }
  python3 -m py_compile "$K/gemm_probe.py" || { echo "gemm_probe.py does not compile"; ok=1; }
  [ -z "$PF_IMG" ] || timeout 120 docker manifest inspect "$PF_IMG" > /dev/null 2>&1 || docker image inspect "$PF_IMG" > /dev/null 2>&1 \
    || { echo "arm image $PF_IMG neither on ghcr nor local"; ok=1; }
  for h in $H1 $H2; do
    x $h "test -d \$HOME/$RC/b12x/compile/preparation" || { echo "$(hn $h): v2.2.0 runtime cache ~/$RC missing"; ok=1; }
    x $h "docker image inspect $SHIP_IMG > /dev/null 2>&1" || { echo "$(hn $h): $SHIP_IMG missing"; ok=1; }
    x $h "docker image inspect $PF_IMG > /dev/null 2>&1" || { [ "${1:-}" = dry ] && echo "note: $(hn $h): $PF_IMG not pulled yet (pulled at run time)" || { echo "$(hn $h): $PF_IMG missing"; ok=1; }; }; done
  for f in "$RD"/*.yaml "$K"/*.sh "$K"/*.py; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  timeout 300 sparkrun registry update $REG1 > /dev/null 2>&1 < /dev/null || echo "note: 1x registry update failed (run retries)"
  preflight dry; rc=$?
  echo "arm image: ${PF_IMG:-none}"
  if [ $rc = 0 ] && x $H1 "docker image inspect $PF_IMG > /dev/null 2>&1" && x $H2 "docker image inspect $PF_IMG > /dev/null 2>&1"; then
    D=$(mktemp -d); env RES="$D" GATE=0 CHAIN_NO_RESTORE=1 bash "$K/td.sh" "$K/tf-01" "$K/tf-02" --dry-run > "$K/td-dry-run.txt" 2>&1; t=$?; rm -rf "$D"
    grep -q REFUSED "$K/td-dry-run.txt" && { echo "td.sh refused an arm ($K/td-dry-run.txt)"; rc=1; }
    [ $t = 0 ] || { echo "td.sh dry-run exit $t ($K/td-dry-run.txt)"; rc=1; }
  else echo "note: thunderdome dry-run skipped until the arm image is on both Sparks"; fi
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: pull arm image"
child timeout 3600 docker pull -q "$PF_IMG" >> "$RES/stage.txt" 2>&1 9>&- || { FINAL="FAILED: pull $PF_IMG on dgx01 ($RES/stage.txt)"; exit 1; }
child timeout 3600 ssh -n -o ConnectTimeout=10 $H2 "docker pull -q $PF_IMG" >> "$RES/stage.txt" 2>&1 9>&- || { FINAL="FAILED: pull $PF_IMG on dgx02 ($RES/stage.txt)"; exit 1; }
[ "$(x $H1 "docker image inspect -f '{{.Id}}' $PF_IMG")" = "$(x $H2 "docker image inspect -f '{{.Id}}' $PF_IMG")" ] || { FINAL="FAILED: arm image Id differs between Sparks"; exit 1; }
st "running: preflight"
timeout 300 sparkrun registry update $REG1 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job"; cp "$K"/run.sh "$K"/hook.sh "$K"/cold.py "$K"/cold_judge.py "$K"/gemm_probe.py "$K"/pool.py "$K"/td.sh "$K/IMAGE" "$RD"/*.yaml "$RES/job/"
x $H1 "docker run --rm --entrypoint cat $PF_IMG /workspace/b12x-source-commit" > "$RES/pf-b12x-commit.txt" 2>&1

regard() { # host (as k88)
  local h=$1 pf=$RES/.guard-$(hn $1).pid p
  p=$(cat "$pf" 2>/dev/null); [ -n "$p" ] || p=$([ $h = $H1 ] && echo $G1 || echo $G2)
  kill -0 "$p" 2>/dev/null && return 0
  if [ $h = $H1 ]; then timeout 6h bash -c "$GUARD" >> "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- &
  else timeout 6h ssh -n $H2 "$GUARD" >> "$RES/guard-dgx02.log" 2>&1 9>&- & fi
  echo $! > "$pf"; log "memory guard $(hn $h) restarted"; }
cid() { x $1 "docker ps -q --filter name=sparkrun | head -1"; }
srl() { flock -o "$RES/.sparkrun.lock" "$@"; }
down() { srl timeout -k 30 300 sparkrun stop --all --hosts $1 >> "$RES/$J.log" 2>&1 9>&-; sleep 5
  x $1 "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" > /dev/null 2>&1; }
prebake() { # host try recipe (as k88, recipe as argument)
  local h=$1 d=$RES/prebake-$(basename "$3" .yaml)-$(hn $1)/try$2 s e c f
  mkdir -p "$d"; cp "$3" "$d/recipe.yaml"
  ( cd "$RD" && srl timeout -k 30 900 sparkrun run "$3" --hosts $h --solo --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  until [ "$(health $h)" = 200 ]; do
    c=$(cid $h); [ -n "$c" ] && x $h "docker exec $c cat /tmp/sparkrun_serve.log" > "$d/serve.log.tmp" 2>/dev/null && mv -f "$d/serve.log.tmp" "$d/serve.log"
    f=$(grep -cE 'Engine core initialization failed|EngineCore failed to start' "$d/serve.log" 2>/dev/null)
    if [ $(( $(date +%s) - s )) -gt 120 ] && { [ -z "$c" ] || [ "${f:-0}" -gt 0 ]; }; then
      echo "boot failed after $(( $(date +%s) - s )) s (container $([ -z "$c" ] && echo gone || echo up))" > "$d/FAILED"; down $h; return 1; fi
    [ $(( $(date +%s) - s )) -gt 5400 ] && { echo "boot timeout" > "$d/FAILED"; down $h; return 1; }
    sleep 15; done
  e=$(date +%s); c=$(cid $h); x $h "docker exec $c cat /tmp/sparkrun_serve.log" > "$d/serve.log" 2>/dev/null
  { echo "boot $((e - s)) s, KV pool $KV_PRE bytes, min MemAvailable $(minmem "$RES/guard-$(hn $h).log" $s $e) GiB"
    grep -hoE '(Directly load AOT compilation|Dynamo bytecode transform time)' "$d/serve.log" | sort | uniq -c
    echo "plans measured: $(grep -ohE '\b[0-9]+ measured' "$d/serve.log" | awk '{s+=$1} END {print s+0}')"
    echo "pong: $(pong $h)"; } > "$d/boot.txt" 2>&1
  down $h; }
prebake_loop() { local t; for t in 1 2 3; do prebake $1 $t "$2" && return 0; regard $1; log "prebake $(basename "$2") $(hn $1) try $t failed"; done; return 1; }
seed() { # cache dir name: copy of the v2.2.0 b12x dir on both Sparks
  local h; for h in $H1 $H2; do
    x $h "set -e; B=\$HOME/$RC; rm -rf \"\$B/$1\"; mkdir -p \"\$B/$1\"; cp -a \"\$B/b12x\" \"\$B/$1/\"; du -sh \"\$B/$1\" | cut -f1" \
      > "$RES/seed-$1-$(hn $h).txt" 2>&1 || return 1; done; }
judge() { # screen label -> RULE per Spark in $S, cold verdict in $CJ
  local a s=$1; S=
  for a in $s-01 $s-02; do [ -d "$RES/$a" ] || { S="$S $a=not-run"; continue; }
    python3 "$K/pool.py" "$RES/none" "$RES/$a" "$RES/pooled-$a" > "$RES/pooled-$a.txt" 2>&1
    S="$S $a=$(sed -n 's/^RULE=//p' "$RES/pooled-$a.txt" | head -1)"; done
  python3 "$K/cold.py" report "$RES/$s-01" "$RES/$s-02" > "$RES/cold-report-$s.txt" 2>&1
  python3 "$K/cold_judge.py" "$RES" "$s-01" "$s-02" > "$RES/cold-judge-$s.txt" 2>&1
  CJ=$(sed -n 's/^COLD_ALL=//p' "$RES/cold-judge-$s.txt"); }
screen() { # label recipe cache
  local s=$1 r1 r2 OK rc V
  st "running: $s seed + prebake (6 GiB KV pool, both Sparks)"
  seed "$3" || { log "$s: seed failed"; return 1; }
  prebake_loop $H1 "$RD/$2-prebake.yaml" > "$RES/prebake-$s-dgx01.log" 2>&1 & local q1=$!
  prebake_loop $H2 "$RD/$2-prebake.yaml" > "$RES/prebake-$s-dgx02.log" 2>&1 & local q2=$!
  CH="$q1 $q2"; wait $q1; r1=$?; wait $q2; r2=$?; CH=
  regard $H1; regard $H2
  OK=; [ $r1 = 0 ] && OK="$K/$s-01"; [ $r2 = 0 ] && OK="$OK $K/$s-02"
  rc=-; V="not run: no prebake came up"
  if [ -n "$OK" ]; then
    st "running: $s screen (thunderdome vs v2.2.0)"
    child env RES="$RES" GPU_LOCK_HELD=1 GATE=0 CHAIN_NO_RESTORE=1 timeout -k 600 28800 bash "$K/td.sh" $OK > "$RES/thunderdome-$s.nohup" 2>&1 < /dev/null; rc=$?
    V=$(head -1 "$RES/STATE" 2>/dev/null); fi
  stop_all; regard $H1; regard $H2
  judge $s
  echo "$s: thunderdome exit $rc ($V); per Spark:$S; cold $CJ" >> "$RES/screens.txt"; log "$s: $S cold $CJ"; }

stop_all
st "running: phase 0 GEMM probe (dgx-01, shipped and arm images)"
mkdir -p "$RES/p0/old" "$RES/p0/new"
probe() { # image outdir [cand]
  timeout -k 30 2400 docker run --rm --gpus all --ipc=host --network none --entrypoint python3 -e CUTE_DSL_ARCH=sm_121a \
    -e XDG_CACHE_HOME=/tmp/cache -e B12X_COMPILE_WORKERS=2 -v "$K:/k90:ro" -v "$RES/p0:/out" "$1" /k90/gemm_probe.py run "/out/$2" ${3:-} \
    > "$RES/p0/$2.log" 2>&1 9>&-; }
probe "$SHIP_IMG" old; probe "$PF_IMG" new cand
timeout -k 30 600 docker run --rm --network none --entrypoint python3 -v "$K:/k90:ro" -v "$RES/p0:/out:ro" "$PF_IMG" \
  /k90/gemm_probe.py compare /out/old /out/new > "$RES/p0/compare.txt" 2>&1 9>&-
grep -qx 'CHECK=PASS' "$RES/p0/compare.txt" || { FINAL="FAILED: k90 phase 0 GEMM probe ($RES/p0/compare.txt)"; exit 1; }
# 2x only if a TP=2 projection changes plan: the K3072 N2560 rows (TP=2 out_proj/o_proj) must show the same plan
TP2=$(awk -F'\t' '$2 == 2560 && $3 == 3072 && $7 != $8 {n++} END {print n+0}' "$RES/p0/compare.txt")
screen tf tf k90-tf
if [ "$TP2" = 0 ]; then echo "2x: skipped, no TP=2 projection changes plan (K3072 N2560 lowers as shipped at every probed M; see phase 0)" >> "$RES/screens.txt"
else echo "2x: needed ($TP2 K3072 N2560 rows change plan) but not run by this job" >> "$RES/screens.txt"; fi

st "running: report"
pf() { local t; t=$(find "$1/hook" -path '*prof-k90*' -name '*.json*' 2>/dev/null | sort | head -1)
  [ -n "$t" ] && python3 "$BL/k81/k81an.py" trace "$t" 2>&1 | sed -n '1,22p' || echo "  no trace in $1"; }
{ echo "k90 $(TZ=Europe/Bucharest date '+%F %T %Z'): 1x v2.2.0 on the b12x large-row tile-fix image ($PF_IMG, b12x $(cat "$RES/pf-b12x-commit.txt" 2>/dev/null)) vs shipped"
  echo; echo "== phase 0 GEMM probe through plan_regimes (capacity 8192): shipped image (old) vs arm image (new); cand = forced 128x128/BK64/u0 dense GEMM alone"
  cat "$RES/p0/compare.txt" 2>/dev/null
  echo; cat "$RES/screens.txt"
  for s in tf; do [ -d "$RES/$s-01" ] || [ -d "$RES/$s-02" ] || continue
    for a in $s-01 $s-02; do echo; echo "-- $a ($([ ${a##*-} = 01 ] && echo dgx-01 || echo dgx-02))"
      grep -vE '^(  self|  cross)' "$RES/pooled-$a.txt" 2>/dev/null | sed -n '/^== acceptance/,$p'
      for b in "$RES/$a"/*-p[0-9]; do [ -d "$b" ] && echo "  $(basename "$b") hook: $(tr '\n' ' ' < "$b/hook/facts.txt" 2>/dev/null)"; done; done
    echo; echo "== cold TTFT $s"; cat "$RES/cold-judge-$s.txt" "$RES/cold-report-$s.txt"
    for a in $s-01 $s-02; do for w in ctl arm; do [ -d "$RES/$a/$w-p1" ] || continue
      echo; echo "== profile $a $w-p1 (one cold 16K prompt)"; pf "$RES/$a/$w-p1"; done; done; done
  echo; echo "== prebakes"; for d in "$RES"/prebake-*/try?; do [ -d "$d" ] && echo "-- ${d#$RES/} $(cat "$d/FAILED" 2>/dev/null) $(tr '\n' ' ' < "$d/boot.txt" 2>/dev/null)"; done
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k90.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
SUM=$(sed 's/ (.*); per Spark:/:/' "$RES/screens.txt" | tr '\n' ';')
{ echo "k90 result: the b12x large-row MXFP8 capacity regime (128x128/BK64 without the large-M unroll for live M >= 4096 on wide-N, K >= 4096 projections; a 4095-row regime keeps the shipped plan below 4096; decode unchanged) against the shipped 1x v2.2.0, ctl arm arm ctl on both Sparks. Measure only; nothing ships from this job."
  echo; echo "Screens: \`$SUM\`. RULE PASS/FLAT = no cell from the T=0 probes, pp2048 or tg512 c1/c8 worse than its noise band."
  [ -s "$K/notes.md" ] && { echo; cat "$K/notes.md"; }
  echo; echo '```'; head -c 55000 "$RES/k90.txt"; echo '```'
  echo; echo "Traces stay on dgx-01 in \`$RES\`. Results: {RESULTS_URL}"; } > "$K/comment.md"
grep -qiE "$PRIV" "$K/comment.md" && { mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k90 comment.md matched the private pattern (not posted; $RES)"; exit 1; }
cp "$K/comment.md" "$RES/comment.md"
FINAL="DONE: k90 $SUM ($RES/k90.txt)"
exit 0
