#!/usr/bin/env bash
# k90b-tile-fix-decode-recheck (2026-10-12, 1x #27): k90 cut down to the cells in question. k90 (b12x b45fe867 large-row
# MXFP8 tile fix vs the shipped 1x v2.2.0) had llama-benchy tg512 c8 on dgx-01 at -3.13% against a 2.68% band while
# dgx-02 read +3.01%. Decode should not move: exact-M decode and every M < 4096 keep the shipped plans.
#   recipes  k90's own control and arm, frozen in $K/rec (copied from k90/rec): control = the shipped 1x v2.2.0 registry
#            recipe + --profiler-config (never started), arm = the same on the tile-fix image with its runtime cache
#            k90-tf (prebaked by k90; nothing here reseeds or deletes it). Preflight checks the registry recipe is
#            still v2.2.0 on the shipped image and that base.yaml is it plus the one profiler line.
#   screen   job-local Thunderdome on both Sparks (one arm per Spark): bake: yes (one unmeasured boot of each recipe,
#            warm since k90), then ctl arm arm ctl ctl arm arm ctl (4 boots per side), warm-up, logits capture,
#            llama-benchy pp2048 c1 + tg512 c1/c8 with 5 runs per boot. No probe cells, no hook, no phase 0.
#   judge    per Spark: pool.py (k90's, report = job-local copy without the probe and wall cells) RULE=PASS|FLAT|FAIL
#            (no cell worse than its noise band) plus the boot-level Welch 95% CI of each cell.
# ETA ~1 h 40 min (bake ~10 min, 8 boots x ~11 min) + restore ~6 min.
# Output: $RES/{tf-01,tf-02,pooled-*,...}, k90b.txt; comment.md here + RESULT. Nothing ships from this job.
# Contract (backlog/runner.sh): caller holds both GPU locks; exit 75 untouched while another job is queued/running;
# STATE first line DONE/FAILED; EXIT trap restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | through the runner (JOBS=k90b). Stop: kill -TERM <pid>.
NEEDS="dgx01 dgx02"
set -u
J=k90b; K=$HOME/GEN-AI/backlog/k90b
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k90b-1x-tile-fix-decode-recheck-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, MODEL
REG1=qwen38-flashnext-1x
REGF=$HOME/.cache/sparkrun/registries/$REG1/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
SHIP_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-d3-hf-20261010-21e0b201-5dad364d-warm
PF_IMG=$(cat "$K/IMAGE" 2>/dev/null)
RC=.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923
RD=$K/rec
PRIV=$(cat "$K/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

mkarms() {
  [ "$(head -1 "$REGF" 2>/dev/null)" = "# Release: v2.2.0" ] || { echo "registry 1x recipe is not v2.2.0 ($REGF)"; return 1; }
  [ "$(sed -n 's/^container: *//p' "$REGF")" = "$SHIP_IMG" ] || { echo "registry 1x container is not $SHIP_IMG"; return 1; }
  [ "$(sed '/^#/d' "$REGF" | diff - "$RD/base.yaml" | grep -c '^[<>]')" = 1 ] || { echo "rec/base.yaml is not the registry recipe + 1 profiler line"; return 1; }
  [ "$(diff "$RD/base.yaml" "$RD/tf.yaml" | grep -c '^>')" = 4 ] && grep -qx "container: $PF_IMG" "$RD/tf.yaml" || { echo "rec/tf.yaml is not k90's arm"; return 1; }
  local n a; for n in 01 02; do a=$K/tf-$n; mkdir -p "$a"
    { echo "# k90b arm on dgx-$n: k90's tile-fix arm (own cache k90-tf, prebaked by k90) vs k90's control"
      echo "name: tf-$n"; echo "recipe: $RD/tf.yaml"; echo "base: $RD/base.yaml"; echo "bake: yes"; } > "$a/thunderdome.arm"; done; }
mktd() { # job-local thunderdome: repo root, ORDER (4 per side), no probe cells, benchy 5 runs; each edit exactly once
  sed -e '/^R=\$(cd "\$(dirname /c\R='"$R" \
      -e 's|^ORDER="ctl:1 arm:1 arm:2 ctl:2"$|ORDER="ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4"|' \
      -e '/^CELLS="fresh-c1/,/count-c8:0:8:2:0:300:20"$/c\CELLS=""' \
      -e 's|--concurrency 1 8 --runs 2 |--concurrency 1 8 --runs 5 |' \
      -e '/^    echo "example   timeout \$t /,+1c\    : ;;' "$TD" > "$K/td.sh"   # the dry-run example line reads a probe cell
  [ "$(diff "$TD" "$K/td.sh" | grep -c '^>')" = 5 ] && ! grep -q '^    echo "example ' "$K/td.sh" && grep -qx "R=$R" "$K/td.sh" && grep -qx 'CELLS=""' "$K/td.sh" \
    && grep -q -- '--runs 5 ' "$K/td.sh" && grep -qx 'ORDER="ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4"' "$K/td.sh" && bash -n "$K/td.sh" \
    || { echo "td.sh edits did not apply"; return 1; }
  # report without probe/wall cells (none are run); pool.py calls this copy
  sed -e 's/^PROBES = ("fresh-c1", "fresh-c4", "fresh-c8", "d16k-c4", "count-c8")$/PROBES = ()/' -e 's/^WALL = ("d16k-c8",)$/WALL = ()/' \
    "$R/scripts/thunderdome_report.py" > "$K/td_report_k90b.py"
  sed "s|^REPORT = .*|REPORT = \"$K/td_report_k90b.py\"|" "$BL/k90/pool.py" > "$K/pool.py"
  ln -sf "$R/scripts/paired_decode_ab.py" "$K/paired_decode_ab.py"   # the report imports it from its own dir
  [ "$(diff "$R/scripts/thunderdome_report.py" "$K/td_report_k90b.py" | grep -c '^>')" = 2 ] && python3 -m py_compile "$K/td_report_k90b.py" \
    && [ "$(diff "$BL/k90/pool.py" "$K/pool.py" | grep -c '^>')" = 1 ] || { echo "report/pool edits did not apply"; return 1; }; }
preflight() {
  local ok=0 h f
  for f in "$TD" "$R/scripts/thunderdome_report.py" "$BL/k90/pool.py" "$RD/base.yaml" "$RD/tf.yaml"; do [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  mkarms || ok=1
  mktd || ok=1
  python3 "$K/pool.py" --selftest > /dev/null || { echo "pool.py selftest failed"; ok=1; }
  for h in $H1 $H2; do
    x $h "test -d \$HOME/$RC/k90-tf/b12x/compile/preparation && test -d \$HOME/$RC/k90-tf/vllm" || { echo "$(hn $h): k90 arm cache ~/$RC/k90-tf missing"; ok=1; }
    x $h "docker image inspect $SHIP_IMG > /dev/null 2>&1" || { echo "$(hn $h): $SHIP_IMG missing"; ok=1; }
    x $h "docker image inspect $PF_IMG > /dev/null 2>&1" || { echo "$(hn $h): $PF_IMG missing"; ok=1; }; done
  for f in "$RD"/*.yaml "$K"/*.sh "$K"/*.py; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  timeout 300 sparkrun registry update $REG1 > /dev/null 2>&1 < /dev/null || echo "note: 1x registry update failed (run retries)"
  preflight; rc=$?
  echo "arm image: ${PF_IMG:-none}"
  if [ $rc = 0 ]; then
    D=$(mktemp -d); env RES="$D" GATE=0 CHAIN_NO_RESTORE=1 bash "$K/td.sh" "$K/tf-01" "$K/tf-02" --dry-run > "$K/td-dry-run.txt" 2>&1; t=$?; rm -rf "$D"
    grep -q REFUSED "$K/td-dry-run.txt" && { echo "td.sh refused an arm ($K/td-dry-run.txt)"; rc=1; }
    [ $t = 0 ] || { echo "td.sh dry-run exit $t ($K/td-dry-run.txt)"; rc=1; }
    grep -E '^(dgx0[12]|per boot|screen) ' "$K/td-dry-run.txt"; fi
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: preflight"
timeout 300 sparkrun registry update $REG1 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job"; cp "$K"/run.sh "$K"/pool.py "$K"/td.sh "$K"/td_report_k90b.py "$K/IMAGE" "$RD"/*.yaml "$RES/job/"

stop_all
st "running: screen (bake + ctl arm arm ctl ctl arm arm ctl on both Sparks)"
child env RES="$RES" GPU_LOCK_HELD=1 GATE=0 CHAIN_NO_RESTORE=1 timeout -k 600 28800 bash "$K/td.sh" "$K/tf-01" "$K/tf-02" > "$RES/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
V=$(head -1 "$RES/STATE" 2>/dev/null)
stop_all
st "running: judge"
S=; for a in tf-01 tf-02; do
  [ -d "$RES/$a" ] || { S="$S $a=not-run"; continue; }
  python3 "$K/pool.py" "$RES/none" "$RES/$a" "$RES/pooled-$a" > "$RES/pooled-$a.txt" 2>&1
  r=$(sed -n 's/^RULE=//p' "$RES/pooled-$a.txt" | head -1)
  grep -q '^VERDICT=' "$RES/pooled-$a.txt" || r="FAIL(no-report)"   # pool.py says FLAT on an empty report
  S="$S $a=$r"; done
# the band is range-based and can widen with more boots: also flag cells whose boot-level 95% CI lies below zero
CIL=; for a in tf-01 tf-02; do
  c=$(sed -n '/^== boot-level Welch/,/^==/p' "$RES/pooled-$a.txt" 2>/dev/null | awk -F'CI \\[' 'NF > 1 { split($2, x, /[],]/); if (x[2] + 0 < 0) { sub(/^ +/, "", $1); sub(/ +[-+][0-9.]+% *$/, "", $1); printf "%s;", $1 } }')
  [ -n "$c" ] && CIL="$CIL $a:${c%;}"; done
echo "thunderdome exit $rc ($V); per Spark:$S; CI below zero:${CIL:- none}" > "$RES/screens.txt"
case "$S" in *=FAIL*|*=not-run*|*"= "*|*=) RU=FAIL ;; *) RU=PASS ;; esac   # job rule: no cell worse beyond noise on either Spark

{ echo "k90b $(TZ=Europe/Bucharest date '+%F %T %Z'): 1x v2.2.0 on the tile-fix image ($PF_IMG) vs shipped, decode recheck, 4 boots per side per Spark"
  echo; cat "$RES/screens.txt"
  for a in tf-01 tf-02; do echo; echo "-- $a ($([ ${a##*-} = 01 ] && echo dgx-01 || echo dgx-02))"
    grep -vE '^(  self|  cross)' "$RES/pooled-$a.txt" 2>/dev/null | sed -n '/^== cells/,$p'; done
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k90b.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
SUM=$(echo "$S" | sed 's/^ //')
{ echo "k90b result: I reran the three llama-benchy cells of k90 (pp2048 c1, tg512 c1, tg512 c8) for the b12x large-row tile fix against the shipped 1x v2.2.0, with 4 boots per side on each Spark in the order ctl arm arm ctl ctl arm arm ctl (k90 had 2 per side). Same recipes and caches as k90. Measure only; nothing ships from this job."
  echo; echo "Per Spark: \`$SUM\`. PASS/FLAT means no cell is worse than its noise band (control boot-to-boot range or within-boot spread, 1% floor). The Welch lines are the boot-level 95% CI of arm vs control."
  echo; case $RU in
    PASS) if [ -n "$CIL" ]; then echo "No cell is worse than its noise band, but the boot-level 95% CI lies entirely below zero for$CIL, so I do not count that as cleared."
          else echo "No cell is worse beyond noise on either Spark and no boot-level CI lies below zero, so the k90 tg512 c8 reading on dgx-01 did not reproduce."; fi ;;
    *) echo "At least one cell is still worse beyond noise (or a Spark did not finish), so the tile fix does not meet my rule yet." ;; esac
  echo; echo '```'; head -c 55000 "$RES/k90b.txt"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
# public-text guard: the comment is posted raw
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k90b comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
FINAL="DONE: k90b RULE=$RU $SUM; CI below zero:${CIL:- none} ($RES/k90b.txt)"
exit 0
