#!/usr/bin/env bash
# k105-1x-gdn-views-c1-recheck (2026-10-16, 1x #65, review of k101s item 1): focused c1 decode recheck of GDN prefill
# views. k101s dgx-01 llama-benchy tg512 c1 was -5.91% with a boot-level 95% CI of [-10.31, -1.50]; the old rule called
# it flat because it sat inside the 6.27% control band. This job isolates the views flag on one image:
#   arms     both sides run k103's arm image ($K3/IMAGE: b12x a1ad1b69 + vLLM 10d0acc8 over 1x v2.3.0) and k103's arm
#            recipe (rec/cb.yaml) without the profiler flag; control VLLM_GDN_PREFILL_VIEWS=0, arm =1; the two recipes
#            differ in that line and the name only. Model, plans, KV pool, cache state unchanged.
#   cache    one runtime cache k105-cb for both sides (the flag is in vllm envs.py ignored_factors: same compile key),
#            a copy of k103-cb (else of the v2.3.0 cache) so k103g's cache stays untouched; bake: yes boots each side
#            once before the screen.
#   screen   job-local Thunderdome on both Sparks, 6 boots per side per Spark, interleaved
#            ctl arm arm ctl ctl arm arm ctl ctl arm arm ctl; per boot: warm-up, fresh-c1 probe x8 (T=0, paired
#            prompts, step gaps), logits a/b, llama-benchy pp2048 + tg512 c1 x5 with the MTP counters read before and
#            after the benchy phase (single concurrency: benchy labels the cells "pp2048" / "tg512" without "(c1)"), nvidia-smi 1 Hz (SM clock, temperature, power, event reasons) during the boot.
#   judge    judge.py per Spark: LOSS if the paired c1 tok/s CI or the benchy tg512 c1 boot CI lies wholly below -1%;
#            PASS if neither does and every measurement is there; missing measurements -> FAILED.
#            STATE: DONE: k105 RULE=PASS|LOSS ... (k103g requires RULE=PASS).
# ETA ~1 h 50 min (seed ~1, bake 2 boots ~6, 12 boots x ~7-8 min per Spark in parallel) + restore ~6 min.
# Output: $RES/{v-01,v-02,seed-*,judge}, k105.txt; comment.md here + RESULT. Nothing ships.
# Contract (backlog/runner.sh): both GPU locks (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE
# first line DONE/SKIPPED/FAILED; EXIT trap restores the shipped 2x (lib.sh).
NEEDS="dgx01 dgx02"
set -u
J=k105; K=$HOME/GEN-AI/backlog/k105; K3=$HOME/GEN-AI/backlog/k103
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k105-1x-gdn-views-c1-recheck-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, MODEL
ARM_IMG=$(cat "$K3/IMAGE" 2>/dev/null); SRC=$K3/rec/cb.yaml
B12X_SHA=a1ad1b69d491333c00c603b10d1cf46f8bf8fc22; VLLM_SHA=10d0acc85fb450472230558eda063c9059c20234
RC=.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923   # 1x model_key (k81)
SHIP_VCR=vllm-b12x-b45fe867; SRCC=k103-cb; ACACHE=k105-cb
CORPUS=$R/results/corpus-code.txt; RD=$K/rec; A1=v-01; A2=v-02
PRIVF=$BL/k98/private.pat; PRIV=$(cat "$PRIVF" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $PRIVF" >&2; exit 2; }

mkrecs() { # on.yaml = k103 arm recipe without the profiler flag, cache k105-cb; off.yaml = the same with views 0
  mkdir -p "$RD"
  [ -s "$SRC" ] || { echo "missing $SRC (k103 writes it)"; return 1; }
  [ -n "$ARM_IMG" ] && grep -qx "container: $ARM_IMG" "$SRC" || { echo "$SRC does not run $ARM_IMG"; return 1; }
  grep -v -- "--profiler-config" "$SRC" | sed -e 's|^name: .*|name: qwen3.8-flash-next-1x-k105-views-on|' \
    -e "s|/cache/runtime/$SRCC\"|/cache/runtime/$ACACHE\"|; s|/cache/runtime/$SRCC/vllm\"|/cache/runtime/$ACACHE/vllm\"|" > "$RD/on.yaml"
  sed -e 's|^name: .*|name: qwen3.8-flash-next-1x-k105-views-off|' \
      -e 's|^  VLLM_GDN_PREFILL_VIEWS: "1"$|  VLLM_GDN_PREFILL_VIEWS: "0"|' "$RD/on.yaml" > "$RD/off.yaml"
  [ "$(diff "$SRC" "$RD/on.yaml" | grep -c '^<')" = 4 ] && [ "$(diff "$SRC" "$RD/on.yaml" | grep -c '^>')" = 3 ] \
    && grep -qx "  XDG_CACHE_HOME: \"/cache/runtime/$ACACHE\"" "$RD/on.yaml" && grep -qx "  VLLM_CACHE_ROOT: \"/cache/runtime/$ACACHE/vllm\"" "$RD/on.yaml" \
    && ! grep -q -- "$SRCC\|--profiler-config" "$RD/on.yaml" && grep -qx '  VLLM_GDN_PREFILL_VIEWS: "1"' "$RD/on.yaml" \
    || { echo "on.yaml edits did not apply"; return 1; }
  [ "$(diff "$RD/on.yaml" "$RD/off.yaml" | grep -c '^>')" = 2 ] && grep -qx '  VLLM_GDN_PREFILL_VIEWS: "0"' "$RD/off.yaml" \
    || { echo "off.yaml edits did not apply"; return 1; }
  local n; for n in 01 02; do mkdir -p "$K/v-$n"
    { echo "# k105 on dgx-$n: views 0 (control) vs 1 (arm), same image and cache k105-cb"
      echo "name: v-$n"; echo "recipe: $RD/on.yaml"; echo "control_recipe: $RD/off.yaml"; echo "base: $RD/off.yaml"; echo "bake: yes"; } > "$K/v-$n/thunderdome.arm"; done; }
mktd() { # job-local thunderdome + report; every edit checked
  sed -e '/^R=\$(cd "\$(dirname /c\R='"$R" \
      -e 's|^ORDER="ctl:1 arm:1 arm:2 ctl:2"$|ORDER="ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4 ctl:5 arm:5 arm:6 ctl:6"|' \
      -e '/^CELLS="fresh-c1/,/count-c8:0:8:2:0:300:20"$/c\CELLS="fresh-c1:256:1:8:104729:600:150"' \
      -e 's|--concurrency 1 8 --runs 2 |--concurrency 1 --runs 5 |' \
      -e 's|^  local h=\$1 d=\$2 rc$|  local h=$1 d=$2 rc; pos "$h" > "$d/pos-benchy.before"|' \
      -e 's|^  log "\$(label "\$h")/\$(basename "\$d"): benchy exit=\$rc"; }$|  pos "$h" > "$d/pos-benchy.after"; log "$(label "$h")/$(basename "$d"): benchy exit=$rc"; }|' \
      -e '/^MEMLOOP=/a SMIQ="nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv,noheader -l 1"' \
      -e '/^MEMLOOP=/a smilog() { if [ "$1" = "$H1" ]; then bash -c "exec $SMIQ" > "$2" 2>&1 < /dev/null 9>&- & else ssh -n "$1" "exec $SMIQ" > "$2" 2>&1 9>&- & fi; echo $!; }' \
      -e 's|^  MP=\$(memlog "\$h" "\$d/mem.log")|  MP=$(memlog "$h" "$d/mem.log"); SM=$(smilog "$h" "$d/smi.csv")|' \
      -e 's|^  kill "\$MP" 2>/dev/null; MP=; }$|  kill "$MP" "$SM" 2>/dev/null; MP=; SM=; }|' \
      -e "s|^  trap 'kill \$MP 2>/dev/null; exit 143' TERM\$|  trap 'kill \$MP \$SM 2>/dev/null; exit 143' TERM|" \
      -e 's|^MODE=screen; MP=$|MODE=screen; MP=; SM=|' \
      -e 's|if \[ "\$item" = arm:1 \]; then python3|if [ "$item" = arm:never ]; then python3|' \
      -e 's|"\$R/scripts/thunderdome_report.py"|"'"$K"'/td_report_k105.py"|g' "$TD" > "$K/td.sh"
  [ "$(diff "$TD" "$K/td.sh" | grep -c '^>')" = 14 ] && grep -qx "R=$R" "$K/td.sh" && grep -q -- '--concurrency 1 --runs 5 ' "$K/td.sh" \
    && grep -q 'pos-benchy.before' "$K/td.sh" && grep -q 'pos-benchy.after' "$K/td.sh" && grep -q '^smilog() ' "$K/td.sh" \
    && grep -q 'SM=$(smilog "$h" "$d/smi.csv")' "$K/td.sh" && grep -q 'kill "$MP" "$SM"' "$K/td.sh" && grep -q 'arm:never' "$K/td.sh" \
    && [ "$(grep -c "$K/td_report_k105.py" "$K/td.sh")" = 2 ] && ! grep -q 'scripts/thunderdome_report.py"' "$K/td.sh" \
    && grep -qx 'CELLS="fresh-c1:256:1:8:104729:600:150"' "$K/td.sh" && bash -n "$K/td.sh" || { echo "td.sh edits did not apply"; return 1; }
  sed -e 's/^PROBES = ("fresh-c1", "fresh-c4", "fresh-c8", "d16k-c4", "count-c8")$/PROBES = ("fresh-c1",)/' \
      -e 's/^WALL = ("d16k-c8",)$/WALL = ()/' \
      -e 's/^BENCHY = ("pp2048 (c1)", "tg512 (c1)", "tg512 (c8)")$/BENCHY = ("pp2048", "tg512")/' \
    "$R/scripts/thunderdome_report.py" > "$K/td_report_k105.py"
  ln -sf "$R/scripts/paired_decode_ab.py" "$K/paired_decode_ab.py"   # report and judge import it from their own dir
  [ "$(diff "$R/scripts/thunderdome_report.py" "$K/td_report_k105.py" | grep -c '^>')" = 3 ] && python3 -m py_compile "$K/td_report_k105.py" \
    || { echo "report edits did not apply"; return 1; }; }
img_ok() { # host: arm image there with both commits
  x $1 "docker run --rm --entrypoint sh $ARM_IMG -c 'cat /workspace/b12x-source-commit; cat /workspace/vllm-overlay-commit'" 2>/dev/null \
    | tr '\n' ' ' | grep -qx "$B12X_SHA $VLLM_SHA "; }
preflight() { # $1 = dry: the arm image may still be pulled at run time
  local ok=0 h f
  for f in "$TD" "$R/scripts/thunderdome_report.py" "$R/scripts/paired_decode_ab.py" "$R/scripts/depth_decode_probe.py" "$CORPUS" "$K/judge.py"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  mkrecs || ok=1
  mktd || ok=1
  python3 "$K/judge.py" --selftest > /dev/null || { echo "judge.py selftest failed"; ok=1; }
  python3 "$K/paired_decode_ab.py" --selftest > /dev/null || { echo "paired_decode_ab.py selftest failed"; ok=1; }
  grep -q '"VLLM_GDN_PREFILL_VIEWS",' "$K3/../k101/overlay/vllm/envs.py" 2>/dev/null \
    && sed -n '/ignored_factors: set\[str\] = {/,/^    }/p' "$K3/../k101/overlay/vllm/envs.py" | grep -q '"VLLM_GDN_PREFILL_VIEWS"' \
    || { echo "VLLM_GDN_PREFILL_VIEWS is not in envs.py ignored_factors of the k101 overlay: one cache for both sides is not valid"; ok=1; }
  for h in $H1 $H2; do
    x $h "test -d \$HOME/$RC/$SRCC/vllm || test -d \$HOME/$RC/$SHIP_VCR" || { echo "$(hn $h): neither ~/$RC/$SRCC nor the v2.3.0 cache"; ok=1; }
    if x $h "docker image inspect $ARM_IMG > /dev/null 2>&1"; then img_ok $h || { echo "$(hn $h): $ARM_IMG does not carry b12x $B12X_SHA + vLLM $VLLM_SHA"; ok=1; }
    else [ "${1:-}" = dry ] && echo "note: $(hn $h): $ARM_IMG not pulled yet (pulled at run time)" || { echo "$(hn $h): $ARM_IMG missing"; ok=1; }; fi; done
  for f in "$RD"/*.yaml "$K"/*.sh "$K"/*.py; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  preflight dry; rc=$?
  if [ $rc = 0 ] && x $H1 "docker image inspect $ARM_IMG > /dev/null 2>&1" && x $H2 "docker image inspect $ARM_IMG > /dev/null 2>&1"; then
    D=$(mktemp -d); env RES="$D" GATE=0 CHAIN_NO_RESTORE=1 bash "$K/td.sh" "$K/$A1" "$K/$A2" --dry-run > "$K/td-dry-run.txt" 2>&1; t=$?; rm -rf "$D"
    grep -q REFUSED "$K/td-dry-run.txt" && { echo "td.sh refused an arm ($K/td-dry-run.txt)"; rc=1; }
    [ $t = 0 ] || { echo "td.sh dry-run exit $t ($K/td-dry-run.txt)"; rc=1; }
  else echo "note: thunderdome dry-run skipped until the arm image is on both Sparks"; fi
  echo "recipes on vs off: $(diff "$RD/on.yaml" "$RD/off.yaml" | grep '^[<>]' | tr '\n' ' ')"
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
mkdir -p "$RES/pull" "$RES/job"
st "running: arm image on both Sparks"
for h in $H1 $H2; do x $h "docker image inspect $ARM_IMG > /dev/null 2>&1 || timeout 3600 docker pull -q $ARM_IMG" \
  >> "$RES/pull/pull.txt" 2>&1 9>&- || { FINAL="FAILED: pull $ARM_IMG on $(hn $h) ($RES/pull)"; exit 1; }; done
[ "$(x $H1 "docker image inspect -f '{{.Id}}' $ARM_IMG")" = "$(x $H2 "docker image inspect -f '{{.Id}}' $ARM_IMG")" ] || { FINAL="FAILED: arm image Id differs between Sparks"; exit 1; }
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp -r "$K"/run.sh "$K"/judge.py "$K"/td.sh "$K"/td_report_k105.py "$RD"/*.yaml "$RES/job/" 2>/dev/null

stop_all
st "running: seed k105-cb (copy of $SRCC, else of the v2.3.0 cache) on both Sparks"
for h in $H1 $H2; do
  x $h "set -e; B=\$HOME/$RC; [ -d \"\$B/$ACACHE/vllm\" ] && { echo kept; exit 0; }; rm -rf \"\$B/$ACACHE\"
    if [ -d \"\$B/$SRCC/vllm\" ]; then cp -a \"\$B/$SRCC\" \"\$B/$ACACHE\"; echo from $SRCC
    else mkdir -p \"\$B/$ACACHE\"; cp -a \"\$B/b12x\" \"\$B/$ACACHE/\"; cp -a \"\$B/$SHIP_VCR\" \"\$B/$ACACHE/vllm\"; echo from v2.3.0; fi
    du -sh \"\$B/$ACACHE\" | cut -f1" > "$RES/seed-$(hn $h).txt" 2>&1 || { FINAL="FAILED: k105 seed on $(hn $h) ($RES/seed-$(hn $h).txt)"; exit 1; }; done

st "running: screen (bake, then ctl arm arm ctl x3 on both Sparks; views 0 vs 1)"
child env RES="$RES" GPU_LOCK_HELD=1 GATE=0 CHAIN_NO_RESTORE=1 timeout -k 600 14400 bash "$K/td.sh" "$K/$A1" "$K/$A2" > "$RES/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
V=$(head -1 "$RES/STATE" 2>/dev/null)
stop_all

st "running: judge"
python3 "$K/judge.py" "$RES/$A1" "$RES/$A2" > "$RES/judge.txt" 2>&1
RU=$(sed -n 's/^RULE=//p' "$RES/judge.txt" | tail -1); SUM=$(sed -n 's/^SPARKS: //p' "$RES/judge.txt" | tail -1)
{ echo "k105 $(TZ=Europe/Bucharest date '+%F %T %Z'): VLLM_GDN_PREFILL_VIEWS=0 vs 1 on $ARM_IMG, same recipe and cache, 6 boots per side per Spark"
  echo "RULE=${RU:-EMPTY}  per Spark: $SUM  (thunderdome exit $rc: $V)"; echo
  cat "$RES/judge.txt"
  for a in $A1 $A2; do echo; echo "-- $a Thunderdome report (fresh-c1 + benchy c1; reported, the rule is judge.py)"; cat "$RES/$a/verdict.txt" 2>/dev/null; done
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k105.txt"

mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
{ echo "k105 result: I reran single-Spark c1 decode with GDN prefill views off and on, on the same image (the k103 candidate), the same recipe, KV pool and runtime cache; only \`VLLM_GDN_PREFILL_VIEWS\` differs (0 vs 1). 6 boots per side on each Spark, interleaved ctl arm arm ctl. Measured per boot: paired c1 decode on fixed prompts (T=0), llama-benchy tg512 c1, MTP acceptance during the benchy phase, GPU clocks and temperature."
  echo; echo "Why: in k101s the dgx-01 tg512 c1 cell was -5.91% with a boot-level 95% CI of [-10.31, -1.50]. My old rule called it flat because it sat inside the 6.27% control band. The rule here: LOSS if the paired c1 CI or the tg512 c1 CI lies wholly below -1% on either Spark."
  echo; echo "\`RULE=${RU:-EMPTY}\`. Per Spark: \`$SUM\`."
  echo; case $RU in
    PASS) echo "Neither c1 measurement is below -1% beyond its interval on either Spark, so the k101 signal did not reproduce with the flag as the only change. The release gate (k103g) may run." ;;
    LOSS) echo "c1 decode is slower with views on beyond the interval on at least one Spark, so the 1x release gate (k103g) does not run on this candidate." ;;
    *) echo "Some measurements are missing, so this run gives no verdict." ;; esac
  echo; echo '```'; head -c 50000 "$RES/k105.txt"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k105 comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
case $RU in PASS|LOSS) FINAL="DONE: k105 RULE=$RU $SUM ($RES/k105.txt)" ;;
  *) FINAL="FAILED: k105 measurements missing: $SUM; $(grep -m3 'missing:' "$RES/judge.txt" | sed 's/^ *//' | tr '\n' ';' | cut -c1-200) ($RES/k105.txt)" ;; esac
exit 0
