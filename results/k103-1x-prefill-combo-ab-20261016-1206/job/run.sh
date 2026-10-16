#!/usr/bin/env bash
# k103-1x-prefill-combo-ab (2026-10-15, 1x #27): the 1x ship candidate with both prefill wins, k100 (HC down projection
# BF16 path, k100ab RULE=WIN) and k101 (GDN prefill views, k101s RULE=WIN), against the shipped 1x v2.3.0.
# Copy of k101s/run.sh (same seed, prebake, Thunderdome screen, hook and judge); only the arm changed:
#   arm      CI image $K/IMAGE (build-b12x-overlay on the 1x v2.3.0 image: b12x a1ad1b69 + vLLM 10d0acc8 files over
#            5dad364d) + VLLM_GDN_PREFILL_VIEWS=1, own runtime cache k103-cb. No local build; the job waits for the
#            image on ghcr (<= 90 min), pulls it on both Sparks, and checks the same Id and both commits in the image.
#   gate     refuses (SKIPPED) unless k100ab STATE says RULE=WIN (else the candidate would be k101 only: queue that by hand).
#   screen   seed + prebake (6 GiB KV, views line on the bake prompt), ctl arm arm ctl ctl arm arm ctl on both Sparks,
#            probes fresh c1/c4/c8, llama-benchy pp2048 c1 + tg512 c1/c4/c8 x3, hook (k101s/hook.sh: cold 16K/64K/128K x3,
#            ident, facts, pass-1 torch profile in prof-k101).
#   judge    as k101s, plus (2026-10-16, review of k101s) a benchy cell whose whole boot-level 95% CI is below -1%
#            counts as worse in pool.py (k101 dgx-01 tg512 c1 was -5.91%, CI [-10.31, -1.50], inside the old 6.27% band).
#            RULE=WIN (no cell worse on either Spark, cold 16K better beyond noise on both, no IDENT FAIL,
#            views on every arm boot only) | LOSS | FLAT; missing measurements -> FAILED. No 2x stage (TP2 off).
#   next     k103g, the release quality gate, runs only on RULE=WIN.
# ETA ~4 h 30 min (pull ~10, seed+prebake ~30, 8 boots x ~25 min per Spark in parallel) + restore ~6 min.
# Output: $RES/{pull,prebake-*,cb-01,cb-02,pooled-*,...}, k103.txt; comment.md here + RESULT. Nothing ships.
# Contract (backlog/runner.sh): both GPU locks (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE
# first line DONE/SKIPPED/FAILED; EXIT trap restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | JOBS="k103 k103g ..." GRACE0=60 setsid nohup bash ~/GEN-AI/backlog/runner.sh > ... &
NEEDS="dgx01 dgx02"
set -u
J=k103; K=$HOME/GEN-AI/backlog/k103; T=$HOME/GEN-AI/backlog/k101   # T: cold/ident/gdnops tools, overlay/VLLM_COMMIT
HOOK=$HOME/GEN-AI/backlog/k101s/hook.sh
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k103-1x-prefill-combo-ab-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, MODEL
REG1=qwen38-flashnext-1x
REGF=$HOME/.cache/sparkrun/registries/$REG1/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
SHIP_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:k95-20261013-b45fe867-5dad364d-warm   # 1x v2.3.0
ARM_IMG=$(cat "$K/IMAGE" 2>/dev/null)
B12X_SHA=a1ad1b69d491333c00c603b10d1cf46f8bf8fc22; VLLM_SHA=10d0acc85fb450472230558eda063c9059c20234
RC=.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923   # 1x model_key (k81)
SHIP_VCR=vllm-b12x-b45fe867; ACACHE=k103-cb; KV_PRE=6442450944
CORPUS=$R/results/corpus-code.txt; RD=$K/rec; A1=cb-01; A2=cb-02
PRIVF=$BL/k98/private.pat; PRIV=$(cat "$PRIVF" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $PRIVF" >&2; exit 2; }

mkrecs() { # base.yaml = registry v2.3.0 + profiler; cb.yaml = arm image, own cache, knob; cb-prebake.yaml = 6 GiB pool
  mkdir -p "$RD"
  [ "$(head -1 "$REGF" 2>/dev/null)" = "# Release: v2.3.0" ] || { echo "registry 1x recipe is not v2.3.0 ($REGF)"; return 1; }
  [ "$(sed -n 's/^container: *//p' "$REGF")" = "$SHIP_IMG" ] || { echo "registry 1x container is not $SHIP_IMG"; return 1; }
  grep -qx "  VLLM_CACHE_ROOT: \"/cache/runtime/$SHIP_VCR\"" "$REGF" || { echo "registry 1x VLLM_CACHE_ROOT is not $SHIP_VCR"; return 1; }
  grep -q '^  VLLM_GDN_PREFILL_VIEWS:' "$REGF" && { echo "registry 1x already sets VLLM_GDN_PREFILL_VIEWS"; return 1; }
  [ -n "$ARM_IMG" ] || { echo "no arm image in $K/IMAGE"; return 1; }
  sed '/^#/d' "$REGF" | sed "s|^    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\\$|    --profiler-config '{{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/runtime/prof-k101\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}}' \\\\\n    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\|" > "$RD/base.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k103-cb|" -e "s|^container: .*|container: $ARM_IMG|" \
      -e "s|^  VLLM_CACHE_ROOT: \"/cache/runtime/$SHIP_VCR\"\$|  XDG_CACHE_HOME: \"/cache/runtime/$ACACHE\"\n  VLLM_CACHE_ROOT: \"/cache/runtime/$ACACHE/vllm\"\n  VLLM_GDN_PREFILL_VIEWS: \"1\"|" \
      "$RD/base.yaml" > "$RD/cb.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k103-cb-prebake|" \
      -e "s|^  kv_cache_memory_bytes: 15032385536\$|  kv_cache_memory_bytes: $KV_PRE|" "$RD/cb.yaml" > "$RD/cb-prebake.yaml"
  [ "$(sed '/^#/d' "$REGF" | diff - "$RD/base.yaml" | grep -c '^>')" = 1 ] || { echo "base.yaml profiler edit did not apply"; return 1; }
  [ "$(diff "$RD/base.yaml" "$RD/cb.yaml" | grep -c '^>')" = 5 ] && grep -qx "container: $ARM_IMG" "$RD/cb.yaml" \
    && grep -qx '  VLLM_GDN_PREFILL_VIEWS: "1"' "$RD/cb.yaml" || { echo "cb.yaml edits did not apply"; return 1; }
  [ "$(diff "$RD/cb.yaml" "$RD/cb-prebake.yaml" | grep -c '^>')" = 2 ] || { echo "cb-prebake.yaml edits did not apply"; return 1; }
  local n a; for n in 01 02; do a=$K/cb-$n; mkdir -p "$a"
    { echo "# k103 arm on dgx-$n: 1x v2.3.0 on the k100+k101 CI image with VLLM_GDN_PREFILL_VIEWS=1, own cache, prebaked"
      echo "name: cb-$n"; echo "recipe: $RD/cb.yaml"; echo "base: $RD/base.yaml"; echo "bake: yes"; echo "hook: $HOOK"; } > "$a/thunderdome.arm"; done; }
mktd() { # job-local thunderdome + report + pool (as k101s); every edit checked
  sed -e '/^R=\$(cd "\$(dirname /c\R='"$R" \
      -e 's|^ORDER="ctl:1 arm:1 arm:2 ctl:2"$|ORDER="ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4"|' \
      -e '/^CELLS="fresh-c1/,/count-c8:0:8:2:0:300:20"$/c\CELLS="fresh-c1:256:1:2:104729:300:45 fresh-c4:256:4:2:314187:400:85 fresh-c8:256:8:2:418916:500:120"' \
      -e 's|timeout -k 30 900 uvx --from "\$BENCHY_SRC" llama-benchy|timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy|' \
      -e 's|--concurrency 1 8 --runs 2 |--concurrency 1 4 8 --runs 3 |' "$TD" > "$K/td.sh"
  [ "$(diff "$TD" "$K/td.sh" | grep -c '^>')" = 5 ] && grep -qx "R=$R" "$K/td.sh" && grep -q -- '--concurrency 1 4 8 --runs 3 ' "$K/td.sh" \
    && grep -q 'timeout -k 30 1800 uvx' "$K/td.sh" && grep -qx 'ORDER="ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4"' "$K/td.sh" && bash -n "$K/td.sh" \
    || { echo "td.sh edits did not apply"; return 1; }
  sed -e 's/^PROBES = ("fresh-c1", "fresh-c4", "fresh-c8", "d16k-c4", "count-c8")$/PROBES = ("fresh-c1", "fresh-c4", "fresh-c8")/' \
      -e 's/^WALL = ("d16k-c8",)$/WALL = ()/' \
      -e 's/^BENCHY = ("pp2048 (c1)", "tg512 (c1)", "tg512 (c8)")$/BENCHY = ("pp2048 (c1)", "tg512 (c1)", "tg512 (c4)", "tg512 (c8)")/' \
    "$R/scripts/thunderdome_report.py" > "$K/td_report_k103.py"
  # pool.py is the job's own file since 2026-10-16 (boot CI guard, review 2026-10-15), no longer a sed copy of k90's
  ln -sf "$R/scripts/paired_decode_ab.py" "$K/paired_decode_ab.py"   # the report imports it from its own dir
  [ "$(diff "$R/scripts/thunderdome_report.py" "$K/td_report_k103.py" | grep -c '^>')" = 3 ] && python3 -m py_compile "$K/td_report_k103.py" \
    && grep -qx "REPORT = \"$K/td_report_k103.py\"" "$K/pool.py" && grep -q '^def ci_guard' "$K/pool.py" \
    && python3 -m py_compile "$K/pool.py" || { echo "report/pool edits did not apply"; return 1; }; }
img_ok() { # host: arm image there with both commits
  x $1 "docker run --rm --entrypoint sh $ARM_IMG -c 'cat /workspace/b12x-source-commit; cat /workspace/vllm-overlay-commit'" 2>/dev/null \
    | tr '\n' ' ' | grep -qx "$B12X_SHA $VLLM_SHA "; }
preflight() { # $1 = dry: the arm image is pulled at run time
  local ok=0 h f
  for f in "$TD" "$R/scripts/thunderdome_report.py" "$R/scripts/paired_decode_ab.py" "$BL/k90/pool.py" "$BL/k81/k81an.py" "$BL/k70/k61.py" \
           "$T/cold.py" "$T/cold_judge.py" "$T/ident.py" "$T/gdnops.py" "$HOOK" "$CORPUS"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  mkrecs || ok=1
  mktd || ok=1
  python3 "$K/pool.py" --selftest > /dev/null || { echo "pool.py selftest failed"; ok=1; }
  head -1 "$T/STATE" 2>/dev/null | grep -q '^DONE: k101 tests=PASS' || { echo "k101 (GPU tests of the views path) is not DONE with tests=PASS"; ok=1; }
  head -1 "$BL/k100/STATE" 2>/dev/null | grep -q '^DONE: k100 RULE=GO' && grep -q 'XCHECK=PASS' "$BL/k100/STATE" || { echo "k100 is not GO + XCHECK=PASS"; ok=1; }
  [ "$(cat "$T/overlay/VLLM_COMMIT" 2>/dev/null)" = "$VLLM_SHA" ] || { echo "k101 VLLM_COMMIT is not $VLLM_SHA"; ok=1; }
  for t in cold.py ident.py gdnops.py; do python3 "$T/$t" selftest > /dev/null || { echo "$t selftest failed"; ok=1; }; done
  python3 "$T/cold_judge.py" --selftest > /dev/null || { echo "cold_judge.py selftest failed"; ok=1; }
  bash -n "$HOOK" || { echo "hook.sh syntax"; ok=1; }
  for h in $H1 $H2; do
    x $h "test -d \$HOME/$RC/b12x/compile/preparation && test -d \$HOME/$RC/$SHIP_VCR" || { echo "$(hn $h): v2.3.0 runtime cache ~/$RC (b12x, $SHIP_VCR) missing"; ok=1; }
    x $h "docker image inspect $SHIP_IMG > /dev/null 2>&1" || { echo "$(hn $h): $SHIP_IMG missing"; ok=1; }
    if x $h "docker image inspect $ARM_IMG > /dev/null 2>&1"; then img_ok $h || { echo "$(hn $h): $ARM_IMG does not carry b12x $B12X_SHA + vLLM $VLLM_SHA"; ok=1; }
    else [ "${1:-}" = dry ] && echo "note: $(hn $h): $ARM_IMG not pulled yet (pulled at run time)" || { echo "$(hn $h): $ARM_IMG missing"; ok=1; }; fi; done
  for f in "$RD"/*.yaml "$K"/*.sh "$K"/*.py "$T"/*.py "$HOOK"; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }
ghcr_wait() { # the CI overlay build pushes the arm image; wait up to 90 min for it on ghcr before any pull (k100)
  local t0=$(date +%s); until timeout 120 docker manifest inspect "$ARM_IMG" > /dev/null 2>&1; do
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && return 1; sleep 60; done; }
win100() { head -1 "$BL/k100ab/STATE" 2>/dev/null | grep -q '^DONE: k100ab RULE=WIN'; }

if [ "${1:-}" = --dry-run ]; then
  timeout 300 sparkrun registry update $REG1 > /dev/null 2>&1 < /dev/null || echo "note: 1x registry update failed (run retries)"
  preflight dry; rc=$?
  win100 && echo "gate: k100ab RULE=WIN (combo arm)" || echo "note: k100ab is not RULE=WIN: the job ends SKIPPED"
  timeout 120 docker manifest inspect "$ARM_IMG" > /dev/null 2>&1 && echo "arm image on ghcr: $ARM_IMG" || echo "note: $ARM_IMG not on ghcr yet (the job waits <= 90 min)"
  if [ $rc = 0 ] && x $H1 "docker image inspect $ARM_IMG > /dev/null 2>&1" && x $H2 "docker image inspect $ARM_IMG > /dev/null 2>&1"; then
    D=$(mktemp -d); env RES="$D" GATE=0 CHAIN_NO_RESTORE=1 bash "$K/td.sh" "$K/$A1" "$K/$A2" --dry-run > "$K/td-dry-run.txt" 2>&1; t=$?; rm -rf "$D"
    grep -q REFUSED "$K/td-dry-run.txt" && { echo "td.sh refused an arm ($K/td-dry-run.txt)"; rc=1; }
    [ $t = 0 ] || { echo "td.sh dry-run exit $t ($K/td-dry-run.txt)"; rc=1; }
  else echo "note: thunderdome dry-run skipped until the arm image is on both Sparks"; fi
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
win100 || { echo "SKIPPED: k103, k100ab is not RULE=WIN ($(head -1 "$BL/k100ab/STATE" | cut -c1-80)); queue a k101-only candidate by hand" > "$K/STATE"; exit 0; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
mkdir -p "$RES/pull" "$RES/job"
st "running: pull arm image on both Sparks"
docker image inspect "$ARM_IMG" > /dev/null 2>&1 || { ghcr_wait && child timeout 3600 docker pull -q "$ARM_IMG" >> "$RES/pull/pull.txt" 2>&1 9>&-; } \
  || { FINAL="FAILED: pull $ARM_IMG on dgx01 (not on ghcr after 90 min or pull error; $RES/pull)"; exit 1; }
x $H2 "docker image inspect $ARM_IMG > /dev/null 2>&1" || child timeout 3600 ssh -n -o ConnectTimeout=10 $H2 "docker pull -q $ARM_IMG" \
  >> "$RES/pull/pull.txt" 2>&1 9>&- || { FINAL="FAILED: pull $ARM_IMG on dgx02 ($RES/pull)"; exit 1; }
[ "$(x $H1 "docker image inspect -f '{{.Id}}' $ARM_IMG")" = "$(x $H2 "docker image inspect -f '{{.Id}}' $ARM_IMG")" ] || { FINAL="FAILED: arm image Id differs between Sparks"; exit 1; }
img_ok $H1 || { FINAL="FAILED: $ARM_IMG does not carry b12x $B12X_SHA + vLLM $VLLM_SHA"; exit 1; }
st "running: preflight"
timeout 300 sparkrun registry update $REG1 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp -r "$K"/run.sh "$K"/IMAGE "$K"/*.py "$K"/td.sh "$HOOK" "$T"/*.py "$RD"/*.yaml "$RES/job/" 2>/dev/null

stop_all

regard() { # host (as k90)
  local h=$1 pf=$RES/.guard-$(hn $1).pid p
  p=$(cat "$pf" 2>/dev/null); [ -n "$p" ] || p=$([ $h = $H1 ] && echo $G1 || echo $G2)
  kill -0 "$p" 2>/dev/null && return 0
  if [ $h = $H1 ]; then timeout 12h bash -c "$GUARD" >> "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- &
  else timeout 12h ssh -n $H2 "$GUARD" >> "$RES/guard-dgx02.log" 2>&1 9>&- & fi
  echo $! > "$pf"; log "memory guard $(hn $h) restarted"; }
cid() { x $1 "docker ps -q --filter name=sparkrun | head -1"; }
srl() { flock -o "$RES/.sparkrun.lock" "$@"; }
down() { srl timeout -k 30 300 sparkrun stop --all --hosts $1 >> "$RES/$J.log" 2>&1 9>&-; sleep 5
  x $1 "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" > /dev/null 2>&1; }
prebake() { # host try recipe (k90)
  local h=$1 d=$RES/prebake-$(hn $1)/try$2 s e c f
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
  e=$(date +%s); c=$(cid $h)
  timeout -k 30 600 python3 "$T/cold.py" run --base "http://$h:8000" --corpus "$CORPUS" --lens 16384 --reps 1 --seed k103bake \
    --cskip 3600000 --out "$d/bake16k.json" > "$d/bake16k.txt" 2>&1   # one prompt through the views path before the screen
  x $h "docker exec $c cat /tmp/sparkrun_serve.log" > "$d/serve.log" 2>/dev/null
  { echo "boot $((e - s)) s, KV pool $KV_PRE bytes, min MemAvailable $(minmem "$RES/guard-$(hn $h).log" $s $(date +%s)) GiB"
    grep -hoE '(Directly load AOT compilation|Dynamo bytecode transform time)' "$d/serve.log" | sort | uniq -c
    echo "plans measured: $(grep -ohE '\b[0-9]+ measured' "$d/serve.log" | awk '{s+=$1} END {print s+0}')"
    echo "views lines: $(grep -c 'GDN prefill views: in-order' "$d/serve.log")"
    echo "pong: $(pong $h)"; } > "$d/boot.txt" 2>&1
  down $h; grep -q '^views lines: [1-9]' "$d/boot.txt" || { echo "views path not engaged on the bake prompt" > "$d/FAILED"; return 1; }; }
prebake_loop() { local t; for t in 1 2 3; do prebake $1 $t "$2" && return 0; regard $1; log "prebake $(hn $1) try $t failed"; done; return 1; }
seed() { # copies of the control's b12x and compile cache dirs, both Sparks
  local h; for h in $H1 $H2; do
    x $h "set -e; B=\$HOME/$RC; [ -d \"\$B/$ACACHE/vllm\" ] && { echo kept; exit 0; }; rm -rf \"\$B/$ACACHE\"; mkdir -p \"\$B/$ACACHE\"; cp -a \"\$B/b12x\" \"\$B/$ACACHE/\"; { cp -a \"\$B/$SHIP_VCR\" \"\$B/$ACACHE/vllm\" 2>/dev/null || { rm -rf \"\$B/$ACACHE/vllm\"; mkdir -p \"\$B/$ACACHE/vllm\"; }; }; du -sh \"\$B/$ACACHE\" | cut -f1" \
      > "$RES/seed-$(hn $h).txt" 2>&1 || return 1; done; }

st "running: seed + prebake (6 GiB KV pool, both Sparks)"
seed || { FINAL="FAILED: k103 seed ($RES/seed-*.txt)"; exit 1; }
prebake_loop $H1 "$RD/cb-prebake.yaml" > "$RES/prebake-dgx01.log" 2>&1 & q1=$!
prebake_loop $H2 "$RD/cb-prebake.yaml" > "$RES/prebake-dgx02.log" 2>&1 & q2=$!
CH="$q1 $q2"; wait $q1; r1=$?; wait $q2; r2=$?; CH=
regard $H1; regard $H2
OK=; [ $r1 = 0 ] && OK="$K/$A1"; [ $r2 = 0 ] && OK="$OK $K/$A2"
[ "$r1$r2" = 00 ] || { FINAL="FAILED: k103 prebake (dgx01 $r1, dgx02 $r2; $RES/prebake-*)"; exit 1; }   # a one-Spark verdict is not 4 boots per side on both

st "running: screen (bake + ctl arm arm ctl ctl arm arm ctl on both Sparks)"
child env RES="$RES" GPU_LOCK_HELD=1 GATE=0 CHAIN_NO_RESTORE=1 timeout -k 600 36000 bash "$K/td.sh" $OK > "$RES/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
V=$(head -1 "$RES/STATE" 2>/dev/null)
stop_all; regard $H1; regard $H2

st "running: judge"
prule() { # node dir, pooled txt -> PASS|FLAT|FAIL(acceptance)|FAIL(worse)|EMPTY(...) (k101s)
  local n=$1 f=$2 r w b k=0
  for b in ctl-p1 ctl-p2 ctl-p3 ctl-p4 arm-p1 arm-p2 arm-p3 arm-p4; do
    [ -d "$n/$b" ] && [ ! -e "$n/$b/FAILED" ] && [ ! -e "$n/$b/COLD" ] && [ ! -e "$n/$b/DIED" ] && k=$((k + 1)); done
  grep -q '^VERDICT=KILL' "$f" 2>/dev/null && { echo "FAIL(acceptance)"; return; }
  grep -q '^VERDICT=' "$f" 2>/dev/null || { echo "EMPTY(no-report)"; return; }
  r=$(sed -n 's/^RULE=//p' "$f" | head -1); w=$(sed -n 's/^worse: //p' "$f" | head -1)
  [ "$r" = FAIL ] && [ -n "$w" ] && [ "$w" != none ] && { echo "FAIL(worse)"; return; }
  [ $k = 8 ] || { echo "EMPTY($k/8 boots)"; return; }
  case "$r" in PASS|FLAT) echo "$r" ;; *) echo "EMPTY(rule ${r:-none}, post checks)" ;; esac; }
coldok() { [ -n "$CJ" ] && ! grep -q 'missing boots' "$RES/cold-judge.txt" && [ "$(grep -cE '[0-9]+K: control' "$RES/cold-judge.txt")" -ge 6 ]; }
S=; ID=; VW=; GO=
for a in $A1 $A2; do
  [ -d "$RES/$a" ] || { S="$S $a=not-run"; continue; }
  python3 "$K/pool.py" "$RES/none" "$RES/$a" "$RES/pooled-$a" > "$RES/pooled-$a.txt" 2>&1
  r=$(prule "$RES/$a" "$RES/pooled-$a.txt")
  S="$S $a=$r"
  python3 "$T/ident.py" report "$RES/$a" > "$RES/ident-$a.txt" 2>&1; ID="$ID $a=$(sed -n 's/^IDENT=//p' "$RES/ident-$a.txt")"
  v=$(for b in "$RES/$a"/*-p[0-9]; do printf '%s:%s ' "$(basename "$b")" "$(sed -n 's/^views //p' "$b/hook/facts.txt" 2>/dev/null)"; done)
  echo "$v" | grep -qE 'ctl-p[0-9]+:[1-9]|arm-p[0-9]+:(0| |$)' && VW="$VW $a:BAD($v)" || VW="$VW $a:ok"
  t1=$(find "$RES/$a/ctl-p1/hook" -path '*prof-k101*' -name '*.json*' 2>/dev/null | sort | head -1)
  t2=$(find "$RES/$a/arm-p1/hook" -path '*prof-k101*' -name '*.json*' 2>/dev/null | sort | head -1)
  { [ -n "$t1" ] && [ -n "$t2" ] && python3 "$T/gdnops.py" diff "$t1" "$t2" || echo "no ctl-p1/arm-p1 trace pair"; } > "$RES/gdnops-$a.txt" 2>&1
  GO="$GO $a:$(sed -n 's/^data movement (non-kept rows) arm - ctl: //p' "$RES/gdnops-$a.txt")"
done
python3 "$T/cold.py" report "$RES/$A1" "$RES/$A2" > "$RES/cold-report.txt" 2>&1
python3 "$T/cold_judge.py" "$RES" $A1 $A2 > "$RES/cold-judge.txt" 2>&1
CJ=$(sed -n 's/^COLD_ALL=//p' "$RES/cold-judge.txt")
echo "thunderdome exit $rc ($V); per Spark:$S; cold $CJ; ident:$ID; views:$VW; GDN data movement per 16K:$GO" > "$RES/screens.txt"
case "$S$ID" in *=FAIL*) RU=LOSS ;;
  *not-run*|*EMPTY*|*"= "*|*=) RU=EMPTY ;;
  *) case "$VW" in *BAD*) RU=EMPTY ;;
     *) if ! coldok; then RU=EMPTY; elif grep -q 'WORSE' "$RES/cold-judge.txt"; then RU=LOSS
        elif [ "$CJ" = PASS ]; then RU=WIN; else RU=FLAT; fi ;; esac ;; esac

{ echo "k103 $(TZ=Europe/Bucharest date '+%F %T %Z'): 1x v2.3.0 on $ARM_IMG (b12x ${B12X_SHA:0:8} + vLLM ${VLLM_SHA:0:8}) + VLLM_GDN_PREFILL_VIEWS=1 vs shipped, 4 boots per side per Spark"
  echo "RULE=$RU"; echo; cat "$RES/screens.txt"
  for a in $A1 $A2; do echo; echo "-- $a ($([ ${a##*-} = 01 ] && echo dgx-01 || echo dgx-02))"
    grep -vE '^(  self|  cross)' "$RES/pooled-$a.txt" 2>/dev/null | sed -n '/^== acceptance/,$p'
    cat "$RES/ident-$a.txt" 2>/dev/null
    echo "== GDN data-movement kernels, one cold 16K prompt (ctl-p1 vs arm-p1)"; cat "$RES/gdnops-$a.txt"
    for b in "$RES/$a"/*-p[0-9]; do [ -d "$b" ] && echo "  $(basename "$b") hook: $(tr '\n' ' ' < "$b/hook/facts.txt" 2>/dev/null)"; done; done
  echo; echo "== cold TTFT"; cat "$RES/cold-judge.txt" "$RES/cold-report.txt"
  for a in $A1 $A2; do for w in ctl arm; do
    t=$(find "$RES/$a/$w-p1/hook" -path '*prof-k101*' -name '*.json*' 2>/dev/null | sort | head -1); [ -n "$t" ] || continue
    echo; echo "== profile $a $w-p1 (one cold 16K prompt, k81 classes)"; python3 "$BL/k81/k81an.py" trace "$t" 2>&1 | sed -n '1,22p'; done; done
  echo; echo "== prebakes"; for d in "$RES"/prebake-*/try?; do [ -d "$d" ] && echo "-- ${d#$RES/} $(cat "$d/FAILED" 2>/dev/null) $(tr '\n' ' ' < "$d/boot.txt" 2>/dev/null)"; done
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k103.txt"

mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
SUM=$(echo "$S" | sed 's/^ //')
{ echo "k103 result: I tested the two prefill changes together, the BF16 path for the hyperconnection down projection (b12x \`${B12X_SHA:0:8}\`) and GDN prefill without token gathers (vLLM \`${VLLM_SHA:0:10}\`, \`VLLM_GDN_PREFILL_VIEWS=1\`), in one image built on top of the v2.3.0 image, against the shipped 1x v2.3.0. 4 boots per side on each Spark in the order ctl arm arm ctl ctl arm arm ctl. Each change won its own A/B before this run. Measure only; nothing ships from this job."
  echo; echo "\`RULE=$RU\`. Per Spark: \`$SUM\`. Cold TTFT: \`COLD_ALL=$CJ\`. Quality identity:\`$ID\`. GDN data-movement kernels per cold 16K prompt (arm - control):\`$GO\`."
  echo; case $RU in
    WIN) echo "Cold prefill is faster beyond boot noise at 16K on both Sparks and no decode, probe or acceptance cell is worse. The release quality gate runs next (k103g)." ;;
    LOSS) echo "At least one cell or cold length is worse beyond noise, or a completion changed, so the combined image does not meet my rule." ;;
    FLAT) echo "Nothing is worse beyond noise, but cold 16K is not faster beyond noise on both Sparks in this run." ;;
    *) echo "Some measurements are missing, so this run gives no verdict." ;; esac
  echo; echo '```'; head -c 55000 "$RES/k103.txt"; echo '```'
  echo; echo "Traces stay on dgx-01 in \`$RES\`. Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k103 comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
case $RU in EMPTY) FINAL="FAILED: k103 measurements missing: $SUM; ident:$ID; views:$VW ($RES/k103.txt)" ;;
  *) FINAL="DONE: k103 RULE=$RU $SUM; cold $CJ; ident:$ID ($RES/k103.txt)" ;; esac
exit 0
