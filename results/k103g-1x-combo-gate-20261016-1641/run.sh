#!/usr/bin/env bash
# k103g-1x-combo-gate (2026-10-15, 1x #27): the release quality gate on the k103 arm (1x v2.3.0 + k100 + k101), the
# gate v2.3.0 shipped with (k95 gate A + k95f gate B, k61.gate rules): stragglers c5-c16 with 0 preemptions, tool-use
# hardmode >= 88, TC-45 x5 = 100, fidelity 8k/32k/64k/128k + 128k seeds 11/13 all 20/20 (no near/wrong/explore/no_call).
#   gate     refuses (SKIPPED) unless k103 STATE says DONE with RULE=WIN and (2026-10-16, review of k101s) k105, the
#            focused c1 decode recheck of GDN views, says DONE with RULE=PASS; the SKIPPED line names which one failed.
#   boot     k103's arm recipe (rec/cb.yaml) without the profiler flag, full KV pool, its prebaked cache k103-cb, solo
#            on each Spark at once; must engage the views path (views line in the serve log after the gate).
#   halves   dgx-01: stragglers, hardmode, TC-45; dgx-02: fidelity + seeds. In parallel, outputs in one dir.
#   not here the packaging check of a release image (warm build with build-b0-warm, fresh-user boot with an empty
#            runtime cache): the CI overlay image has no baked AOT seed, so that check belongs to the release build.
# ETA ~2 h (boots ~10, hardmode up to 2 h; fidelity ~50 min in parallel) + restore ~6 min.
# Contract (backlog/runner.sh): both GPU locks; exit 75 while busy; STATE DONE/SKIPPED/FAILED; comment.md + RESULT.
NEEDS="dgx01 dgx02"
set -u
J=k103g; K=$HOME/GEN-AI/backlog/k103g; K3=$HOME/GEN-AI/backlog/k103; K5=$HOME/GEN-AI/backlog/k105
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k103g-1x-combo-gate-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, MODEL
K70=$BL/k70; SRC=$K3/rec/cb.yaml; REC=$K/gate.yaml; CORPUS=$R/results/corpus-code.txt
VIEWS='GDN prefill views: in-order'
PRIV=$(cat "$BL/k98/private.pat" 2>/dev/null); [ -n "$PRIV" ] || { echo "missing $BL/k98/private.pat" >&2; exit 2; }
mkrec() { # gate.yaml = k103's arm recipe without the profiler flag (one line less, nothing else)
  [ -s "$SRC" ] || { echo "missing $SRC (k103 writes it)"; return 1; }
  grep -v -- "--profiler-config" "$SRC" | sed 's|^name: .*|name: qwen3.8-flash-next-1x-k103-gate|' > "$REC"
  [ "$(diff "$SRC" "$REC" | grep -c '^<')" = 2 ] && [ "$(diff "$SRC" "$REC" | grep -c '^>')" = 1 ] \
    && grep -qx '  VLLM_GDN_PREFILL_VIEWS: "1"' "$REC" && grep -q '^container: .*k103-' "$REC" || { echo "gate.yaml edit did not apply"; return 1; }; }
checks() {
  local ok=0 f t
  mkrec || ok=1
  for f in "$K70/k61.py" "$R/scripts/straggler_probe.py" "$R/scripts/fidelity_probe.py"; do [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  for t in tool-eval-bench sparkrun python3; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  return $ok; }
win() { head -1 "$K3/STATE" 2>/dev/null | grep -q '^DONE: k103 RULE=WIN'; }
pass105() { head -1 "$K5/STATE" 2>/dev/null | grep -q '^DONE: k105 RULE=PASS'; }
if [ "${1:-}" = --dry-run ]; then checks; rc=$?
  win && echo "gate: k103 RULE=WIN" || echo "note: k103 is not DONE with RULE=WIN yet ($(head -1 "$K3/STATE" 2>/dev/null | cut -c1-60)): the job ends SKIPPED unless it is by then"
  pass105 && echo "gate: k105 RULE=PASS" || echo "note: k105 is not DONE with RULE=PASS yet ($(head -1 "$K5/STATE" 2>/dev/null | cut -c1-60)): the job ends SKIPPED unless it is by then"
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"; echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
win || { echo "SKIPPED: k103g, k103 is not DONE with RULE=WIN ($(head -1 "$K3/STATE" 2>/dev/null | cut -c1-80))" > "$K/STATE"; exit 0; }
pass105 || { echo "SKIPPED: k103g, k105 (c1 decode recheck of GDN views) is not DONE with RULE=PASS ($(head -1 "$K5/STATE" 2>/dev/null | cut -c1-100))" > "$K/STATE"; exit 0; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
checks > "$RES/checks.txt" 2>&1 || { FINAL="FAILED: k103g checks ($RES/checks.txt)"; exit 1; }
cp "$K/run.sh" "$REC" "$RES/"
stop_all
G=$RES/gate; mkdir -p "$G"
cid() { x $1 "docker ps -q --filter name=sparkrun | head -1"; }
slog() { local c; c=$(cid $1); [ -n "$c" ] && x $1 "docker exec $c cat /tmp/sparkrun_serve.log" > "$2" 2>/dev/null; }
gboot() { # host: full-pool solo boot of gate.yaml, cap 5400 s
  local h=$1 d=$RES/boot-$(hn $1) s f c; mkdir -p "$d"
  ( cd "$K" && flock -o "$RES/.sparkrun.lock" timeout -k 30 900 sparkrun run "$REC" --hosts $h --solo --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  until [ "$(health $h)" = 200 ]; do
    c=$(cid $h); slog $h "$d/serve.log"
    f=$(grep -cE 'Engine core initialization failed|EngineCore failed to start' "$d/serve.log" 2>/dev/null)
    if [ $(( $(date +%s) - s )) -gt 120 ] && { [ -z "$c" ] || [ "${f:-0}" -gt 0 ]; }; then echo "boot failed" > "$d/FAILED"; return 1; fi
    [ $(( $(date +%s) - s )) -gt 5400 ] && { echo "boot timeout" > "$d/FAILED"; return 1; }
    sleep 15; done
  echo "boot $(( $(date +%s) - s )) s; pong $(pong $h)" > "$d/boot.txt"; grep -qi 'pong.*pong' "$d/boot.txt"; }
st "running: gate boots (both Sparks)"
gboot $H1 & b1=$!; gboot $H2 & b2=$!; CH="$b1 $b2"; wait $b1; r1=$?; wait $b2; r2=$?; CH=
[ "$r1$r2" = 00 ] || { FINAL="FAILED: k103g boot (dgx01 $r1, dgx02 $r2; $RES/boot-*)"; exit 1; }
st "running: gate (dgx-01 stragglers, hardmode, TC-45; dgx-02 fidelity 8k-128k + 128k seeds 11/13)"
halfA() {
  ( cd "$R" && timeout 900 python3 scripts/straggler_probe.py 5 6 7 8 12 16 > "$G/straggler.log" 2>&1 )
  timeout 7200 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://$H1:8000 --model $MODEL > "$G/hardmode.log" 2>&1
  timeout 3600 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://$H1:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$G/tc45.log" 2>&1; }
halfB() {
  ( cd "$R" && timeout 5400 python3 scripts/fidelity_probe.py --base http://$H2:8000 --model $MODEL --depths 8000,32000,64000,128000 --out "$G/fidelity.json" > "$G/fidelity_probe.txt" 2>&1 )
  for sd in 11 13; do ( cd "$R" && timeout 2700 python3 scripts/fidelity_probe.py --base http://$H2:8000 --model $MODEL --depths 128000 --seed $sd --out "$G/fidelity-seed$sd.json" > "$G/fidelity-seed$sd.txt" 2>&1 ); done; }
halfA & a1=$!; halfB & a2=$!; CH="$a1 $a2"; wait $a1; wait $a2; CH=
for h in $H1 $H2; do slog $h "$RES/boot-$(hn $h)/serve.log"; done
stop_all
F=; for h in $H1 $H2; do d=$RES/boot-$(hn $h)
  echo "$(hn $h): views lines $(grep -c "$VIEWS" "$d/serve.log"), plans measured $(grep -ohE '\b[0-9]+ measured' "$d/serve.log" | awk '{s+=$1} END {print s+0}'), AOT load $(grep -c 'Directly load AOT compilation' "$d/serve.log"), Traceback $(grep -c Traceback "$d/serve.log")" >> "$RES/facts.txt"
  grep -q "$VIEWS" "$d/serve.log" || F="$F $(hn $h): views path not engaged;"; done
t=$(mktemp -d); ln -s "$G" "$t/cand1"; ln -s "$G" "$t/cand2"
GV=$(python3 -c "import sys; sys.path.insert(0, sys.argv[1]); import k61; b, hm = k61.gate(sys.argv[2]); print('PASS' if not b else 'FAIL (' + '; '.join(b) + ')')" "$K70" "$t" 2>&1); rm -rf "$t"
{ echo "k103g $(TZ=Europe/Bucharest date '+%F %T %Z'): release quality gate on the k103 arm ($(sed -n 's/^container: //p' "$REC"), VLLM_GDN_PREFILL_VIEWS=1)"
  echo "GATE=$GV"; cat "$RES/facts.txt"; cat "$RES"/boot-*/boot.txt
  grep -hE "Quality:" "$G/hardmode.log"; grep -hE "Pass\^5|Score:" "$G/tc45.log" | head -2
  grep -h "^depth" "$G/fidelity_probe.txt" "$G"/fidelity-seed1?.txt; tail -8 "$G/straggler.log"; } > "$RES/k103g.txt" 2>&1
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
{ echo "k103g result: I ran the release quality gate of v2.3.0 on the k103 candidate (v2.3.0 plus the HC down projection BF16 path and GDN prefill views), one boot per Spark: tool use, stragglers and TC-45 on one, long-context retrieval on the other."
  echo; echo "Gate: \`${GV}\`.${F:+ Views path check failed:$F}"
  echo; echo '```'; head -c 30000 "$RES/k103g.txt"; echo '```'
  echo; echo "Before a release the image still needs the warm build and a fresh-user boot check. Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k103g comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
[ -n "$F" ] && { FINAL="FAILED: k103g views path not engaged:$F gate ${GV%% (*} ($RES/k103g.txt)"; exit 1; }
FINAL="DONE: k103g gate=$GV ($RES/k103g.txt)"
exit 0
