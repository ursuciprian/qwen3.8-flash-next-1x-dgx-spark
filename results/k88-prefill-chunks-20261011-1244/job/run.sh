#!/usr/bin/env bash
# k88-prefill-chunks (2026-10-11, 1x #27): the cheap prefill fix from k81, user-approved. Measure only; nothing ships.
# k81 found: with --mamba-cache-mode align every prefill chunk ends on a 3,024-token boundary, so max_num_batched_tokens
# 8192 gives 6,048-token chunks and a 16K prompt takes 4 steps (6048/6048/3024/1264) at ~266 ms fixed per step (the
# routed MoE reads all expert weights once per step); and the MXFP8 dense GEMM K6144 N2560 (GDN out_proj, attention
# o_proj) is 4.7x / 3.4x slower at M 6048 than at M 3024.
#   P0 probe   (dgx-01, in the 1x image, ~15 min, cap 25 min, while dgx-02 prebakes) gemm_probe.py: that GEMM (and in_proj as a
#              linear reference) per M through the serving capacity regime (plan for max_rows 8192 / 9072 / 15120,
#              expected_m None), the same with the large-M unroll forced off, and exact-M plans. Diagnosis only: the
#              capacity regime's dense tile/unroll comes from b12x's default lowering, not from the autotuned (seedable)
#              selection, so a plan fix that the probe points to needs a b12x policy change (a later step, not this job).
#   P1 1x A/B  control = shipped 1x v2.2.0 (registry, header stripped) + --profiler-config (not a compile factor, AOT
#              still loads); arm = the same with max_num_batched_tokens 9072 (3 x 3024: 16K becomes 9072/6048/1264 =
#              3 steps) and its own runtime cache k88-a seeded from the v2.2.0 b12x cache. Arm prebake with a 6 GiB KV
#              pool on both Sparks (k78 lesson), then job-local Thunderdome on both Sparks, ctl arm arm ctl, bake: yes:
#              T=0 probes fresh c1/c4/c8 + d16k-c4, logits capture, llama-benchy pp2048 c1 + tg512 c1/c8 (3 runs), hook:
#              cold TTFT 16K/64K/128K x3 + boot facts + a torch-profiler window on each pass-1 boot (chunking and
#              per-kernel times on ctl and arm). Judge per Spark: pool.py (k80 rule with TritonBundler tracebacks
#              filtered: RULE=PASS|FLAT|FAIL, 0 plans measured, KV equal, AOT loaded, MemAvailable >= 6 GiB) + cold TTFT
#              vs the control's boot-to-boot spread.
#   P2         dropped (coordinator, 2026-10-11 10:50): the user serves on the 2x; a 2x A/B is a separate evening job
#              once a real fix exists (twox.sh stays in the job dir, unused).
# ETA ~2 h 20 min (P0 15 + prebake ~15 + screen ~105 + report) + restore ~6 min.
# Output: $RES (p0/, prebake-*/, mnbt-01/, mnbt-02/, pooled-*, 2x/ ...), k88.txt; comment.md here + RESULT.
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k88; K=$HOME/GEN-AI/backlog/k88
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k88-prefill-chunks-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, MODEL
REG1=qwen38-flashnext-1x
REGF=$HOME/.cache/sparkrun/registries/$REG1/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
SHIP_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:tp1-d3-hf-20261010-21e0b201-5dad364d-warm
RC=.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923   # 1x model_key (k81)
ARMC=k88-a; KV_PRE=6442450944; MNBT=9072; ARMS="mnbt-01 mnbt-02"
CORPUS=$R/results/corpus-code.txt; RD=$K/rec; GATE=0
PRIV=$(cat "$K/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

mkrec1() { # registry 1x v2.2.0 -> base.yaml (+profiler), arm.yaml, prebake.yaml; arm dirs
  mkdir -p "$RD"
  [ "$(head -1 "$REGF" 2>/dev/null)" = "# Release: v2.2.0" ] || { echo "registry 1x recipe is not v2.2.0 ($REGF)"; return 1; }
  [ "$(sed -n 's/^container: *//p' "$REGF")" = "$SHIP_IMG" ] || { echo "registry 1x container is not $SHIP_IMG"; return 1; }
  sed '/^#/d' "$REGF" | sed "s|^    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\\$|    --profiler-config '{{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/runtime/prof-k88\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}}' \\\\\n    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\|" > "$RD/base.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k88-mnbt$MNBT|" \
      -e "s|^  max_num_batched_tokens: 8192\$|  max_num_batched_tokens: $MNBT|" \
      -e "s|^env:\$|env:\n  XDG_CACHE_HOME: \"/cache/runtime/$ARMC\"\n  VLLM_CACHE_ROOT: \"/cache/runtime/$ARMC/vllm\"|" "$RD/base.yaml" > "$RD/arm.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k88-prebake|" \
      -e "s|^  kv_cache_memory_bytes: 15032385536\$|  kv_cache_memory_bytes: $KV_PRE|" "$RD/arm.yaml" > "$RD/prebake.yaml"
  [ "$(sed '/^#/d' "$REGF" | diff - "$RD/base.yaml" | grep -c '^>')" = 1 ] && grep -q 'prof-k88' "$RD/base.yaml" \
    || { echo "base.yaml profiler edit did not apply"; return 1; }
  [ "$(diff "$RD/base.yaml" "$RD/arm.yaml" | grep -c '^>')" = 4 ] && grep -qx "  max_num_batched_tokens: $MNBT" "$RD/arm.yaml" \
    || { echo "arm.yaml edits did not apply"; return 1; }
  [ "$(diff "$RD/arm.yaml" "$RD/prebake.yaml" | grep -c '^>')" = 2 ] && grep -qx "  kv_cache_memory_bytes: $KV_PRE" "$RD/prebake.yaml" \
    || { echo "prebake.yaml edits did not apply"; return 1; }
  local a; for a in $ARMS; do mkdir -p "$K/$a"
    { echo "# k88 arm on $([ $a = mnbt-01 ] && echo dgx-01 || echo dgx-02): 1x v2.2.0 with max_num_batched_tokens $MNBT, own cache seeded from the v2.2.0 b12x cache, prebaked; both sides with --profiler-config"
      echo "name: $a"; echo "recipe: $RD/arm.yaml"; echo "base: $RD/base.yaml"; echo "bake: yes"; echo "hook: $K/hook.sh"; } > "$K/$a/thunderdome.arm"; done; }
mktd() { # job-local thunderdome: fixed repo root, probe cells, benchy runs 3; each edit applies exactly once
  sed -e '/^R=\$(cd "\$(dirname /c\R='"$R" \
      -e '/^CELLS="fresh-c1/,/count-c8:0:8:2:0:300:20"$/c\CELLS="fresh-c1:256:1:2:104729:300:45 fresh-c4:256:4:2:314187:400:85 fresh-c8:256:8:2:418916:500:120 d16k-c4:16384:4:2:1361477:700:150"' \
      -e 's|--concurrency 1 8 --runs 2 |--concurrency 1 8 --runs 3 |' "$TD" > "$K/td.sh"
  local n; n=$(diff "$TD" "$K/td.sh" | grep -c '^>')
  [ "$n" = 3 ] && grep -qx "R=$R" "$K/td.sh" && grep -q -- '--runs 3 ' "$K/td.sh" && grep -q '^CELLS="fresh-c1:256:1:2:104729:300:45 fresh-c4' "$K/td.sh" \
    && bash -n "$K/td.sh" || { echo "td.sh edits did not apply ($n changed lines)"; return 1; }; }
preflight() { # $1 = dry
  local ok=0 h f
  for f in "$TD" "$R/scripts/thunderdome_report.py" "$K/pool.py" "$K/cold.py" "$K/hook.sh" "$K/gemm_probe.py" "$BL/k81/k81an.py" "$CORPUS"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  mkrec1 || ok=1
  mktd || ok=1
  python3 "$K/pool.py" --selftest > /dev/null || { echo "pool.py selftest failed"; ok=1; }
  python3 "$K/cold.py" selftest > /dev/null || { echo "cold.py selftest failed"; ok=1; }
  python3 "$BL/k81/k81an.py" selftest > /dev/null || { echo "k81an.py selftest failed"; ok=1; }
  python3 -m py_compile "$K/gemm_probe.py" || { echo "gemm_probe.py does not compile"; ok=1; }
  bash -n "$K/hook.sh" || { echo "hook.sh syntax"; ok=1; }
  python3 "$R/scripts/thunderdome_report.py" --selftest > /dev/null 2>&1 || { echo "thunderdome_report.py selftest failed"; ok=1; }
  for h in $H1 $H2; do
    x $h "test -d \$HOME/$RC/b12x/compile/preparation" || { echo "$(hn $h): v2.2.0 runtime cache ~/$RC missing"; ok=1; }
    x $h "docker image inspect $SHIP_IMG > /dev/null 2>&1" || { echo "$(hn $h): $SHIP_IMG missing"; ok=1; }; done
  for f in "$RD"/*.yaml "$K"/*.sh "$K"/*.py; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  timeout 300 sparkrun registry update $REG1 > /dev/null 2>&1 < /dev/null || echo "note: 1x registry update failed (run retries)"
  preflight dry; rc=$?
  if [ $rc = 0 ]; then D=$(mktemp -d); env RES="$D" GATE=0 CHAIN_NO_RESTORE=1 bash "$K/td.sh" "$K/mnbt-01" "$K/mnbt-02" --dry-run > "$K/td-dry-run.txt" 2>&1; t=$?; rm -rf "$D"
    grep -q REFUSED "$K/td-dry-run.txt" && { echo "td.sh refused an arm ($K/td-dry-run.txt)"; rc=1; }
    [ $t = 0 ] || { echo "td.sh dry-run exit $t ($K/td-dry-run.txt)"; rc=1; }; fi
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: P0 GEMM probe on dgx-01 || arm prebake on dgx-02, then dgx-01 prebake; Thunderdome mnbt$MNBT vs v2.2.0 on both Sparks; no P2"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: preflight"
timeout 300 sparkrun registry update $REG1 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job"; cp "$K"/run.sh "$K"/hook.sh "$K"/cold.py "$K"/pool.py "$K"/gemm_probe.py "$K"/td.sh "$RD"/*.yaml "$RES/job/"

regard() { # host: restart that Spark's memory guard if it aborted (capped at 6 h; the EXIT trap only knows $G1 $G2)
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
prebake() { # host try (k78): arm recipe with the 6 GiB pool into the seeded arm cache, then stop. Fails fast on a dead engine.
  local h=$1 d=$RES/prebake-$(hn $1)/try$2 s e c f
  mkdir -p "$d"; cp "$RD/prebake.yaml" "$d/recipe.yaml"
  ( cd "$RD" && srl timeout -k 30 900 sparkrun run "$RD/prebake.yaml" --hosts $h --solo --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
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
prebake_loop() { # host -> 0 when a try came up
  local t; for t in 1 2 3; do prebake $1 $t && return 0; regard $1; log "prebake $(hn $1) try $t failed"; done; return 1; }
p0() { # dgx-01: GEMM probe in the shipped 1x image (GPU free: nothing else runs on dgx-01 now)
  mkdir -p "$RES/p0"
  timeout -k 30 1500 docker run --rm --gpus all --ipc=host --network none --entrypoint python3 \
    -e CUTE_DSL_ARCH=sm_121a -e XDG_CACHE_HOME=/tmp/cache -e B12X_COMPILE_WORKERS=2 \
    -v "$K:/k88:ro" -v "$RES/p0:/out" "$SHIP_IMG" /k88/gemm_probe.py /out/gemm_probe.tsv > "$RES/p0/gemm_probe.log" 2>&1
  log "P0 gemm probe exit=$?"; }

st "running: P0 GEMM probe (dgx-01) + 1x arm prebake (dgx-02), stop 2x, seed"
stop_all
for h in $H1 $H2; do
  x $h "set -e; B=\$HOME/$RC; rm -rf \"\$B/$ARMC\"; mkdir -p \"\$B/$ARMC\"; cp -a \"\$B/b12x\" \"\$B/$ARMC/\"; du -sh \"\$B/$ARMC\" | cut -f1" \
    > "$RES/seed-$(hn $h).txt" 2>&1 || { FINAL="FAILED: seed on $(hn $h) ($RES/seed-$(hn $h).txt)"; exit 1; }; done
( p0; prebake_loop $H1 ) > "$RES/seq-dgx01.log" 2>&1 & q1=$!
prebake_loop $H2 > "$RES/seq-dgx02.log" 2>&1 & q2=$!
CH="$q1 $q2"; wait $q1; r1=$?; wait $q2; r2=$?; CH=
regard $H1; regard $H2
OK=; [ $r1 = 0 ] && OK="$K/mnbt-01"; [ $r2 = 0 ] && OK="$OK $K/mnbt-02"
log "prebake dgx01 rc=$r1 dgx02 rc=$r2"

rc=-; V="not run: no prebake came up"
if [ -n "$OK" ]; then
  st "running: P1 screen (thunderdome mnbt$MNBT vs v2.2.0:$(for a in $OK; do printf ' %s' "$(basename "$a")"; done))"
  child env RES="$RES" GPU_LOCK_HELD=1 GATE=0 CHAIN_NO_RESTORE=1 timeout -k 600 28800 bash "$K/td.sh" $OK > "$RES/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
  V=$(head -1 "$RES/STATE" 2>/dev/null); log "thunderdome exit=$rc STATE: $V"; fi
stop_all; regard $H1; regard $H2

st "running: P1 judge"
S=; for a in $ARMS; do [ -d "$RES/$a" ] || { S="$S $a=not-run"; continue; }
  python3 "$K/pool.py" "$RES/none" "$RES/$a" "$RES/pooled-$a" > "$RES/pooled-$a.txt" 2>&1
  S="$S $a=$(sed -n 's/^RULE=//p' "$RES/pooled-$a.txt" | head -1)"; done
python3 "$K/cold.py" report "$RES"/mnbt-0? > "$RES/cold-report.txt" 2>&1
python3 - "$RES" > "$RES/cold-judge.txt" 2>&1 <<'EOF'
# per Spark and length: arm vs control cold tok/s (mean over boots), noise = the control's boot-to-boot spread (>= 1%)
import glob, json, os, statistics, sys
res = sys.argv[1]; ok_all = True
for a in ("mnbt-01", "mnbt-02"):
    per = {}
    for b in sorted(glob.glob(f"{res}/{a}/*-p[0-9]")):
        try: s = json.load(open(f"{b}/hook/cold.json"))["summary"]
        except (OSError, ValueError, KeyError): continue
        for n, v in s.items():
            if v.get("n"): per.setdefault(n, {}).setdefault(os.path.basename(b)[:3], []).append(v["tok_s"])
    ok = bool(per)
    for n in sorted(per, key=int):
        c, m = per[n].get("ctl", []), per[n].get("arm", [])
        if not c or not m: print(f"{a} {int(n)//1024}K: missing boots"); ok = False; continue
        d = 100 * (statistics.mean(m) / statistics.mean(c) - 1)
        noise = max(1.0, 100 * (max(c) - min(c)) / statistics.mean(c)) if len(c) > 1 else 3.0
        tag = "better" if d > noise else "WORSE" if d < -noise else "flat"
        if tag == "WORSE" or (n == "16384" and tag != "better"): ok = False
        print(f"{a} {int(n)//1024:>4}K: control {statistics.mean(c):6.0f} arm {statistics.mean(m):6.0f} tok/s ({d:+.1f}%, noise {noise:.1f}%) {tag}")
    print(f"{a} COLD={'PASS' if ok else 'FAIL'}"); ok_all &= ok
print(f"COLD_ALL={'PASS' if ok_all else 'FAIL'}")
EOF
CJ=$(sed -n 's/^COLD_ALL=//p' "$RES/cold-judge.txt")
P1=FAIL; case "$S" in *mnbt-01=PASS*|*mnbt-01=FLAT*) case "$S" in *mnbt-02=PASS*|*mnbt-02=FLAT*) [ "$CJ" = PASS ] && P1=PASS ;; esac ;; esac
log "P1: $S, cold $CJ -> $P1"

P2="not run (dropped from this job; a 2x A/B follows a real fix)"

st "running: report"
pf() { # boot dir -> chunking and per-class line from its pass-1 profile
  local t; t=$(find "$1/hook" -path '*prof-k88*' -name '*.json*' 2>/dev/null | sort | head -1)
  [ -n "$t" ] && python3 "$BL/k81/k81an.py" trace "$t" 2>&1 | sed -n '1,22p' || echo "  no trace in $1"; }
{ echo "k88 $(TZ=Europe/Bucharest date '+%F %T %Z'): 1x v2.2.0 max_num_batched_tokens $MNBT vs 8192 (both with the profiler config); P1 $P1; P2 $P2"
  echo; echo "== P0 GEMM probe (dgx-01, shipped 1x image): ms per call by M; capN = serving capacity regime for max_rows N"
  grep '^#' "$RES/p0/gemm_probe.tsv" 2>/dev/null | sed -n '/summary/,$p' || echo "  probe failed: $(tail -3 "$RES/p0/gemm_probe.log" 2>/dev/null)"
  grep -c FAILED "$RES/p0/gemm_probe.tsv" 2>/dev/null | sed 's/^/failed probe variants: /'
  echo; echo "== prebake (arm, 6 GiB KV pool)"
  for h in dgx01 dgx02; do for t in "$RES/prebake-$h"/try?; do [ -d "$t" ] || continue
    echo "-- $h $(basename "$t") $(cat "$t/FAILED" 2>/dev/null)"; cat "$t/boot.txt" 2>/dev/null; done; done
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'
  echo; echo "== P1 thunderdome exit $rc ($V); per Spark:$S; cold: $CJ"
  for a in $ARMS; do echo; echo "-- $a ($([ $a = mnbt-01 ] && echo dgx-01 || echo dgx-02))"
    grep -vE '^(  self|  cross)' "$RES/pooled-$a.txt" 2>/dev/null | sed -n '/^== acceptance/,$p'
    for b in "$RES/$a"/*-p[0-9]; do [ -d "$b" ] && echo "  $(basename "$b") hook: $(tr '\n' ' ' < "$b/hook/facts.txt" 2>/dev/null)"; done; done
  echo; echo "== cold TTFT (hook: 3 cold prompts per length per boot, same prompts in every boot)"; cat "$RES/cold-judge.txt"; cat "$RES/cold-report.txt"
  for a in $ARMS; do for w in ctl arm; do [ -d "$RES/$a/$w-p1" ] || continue
    echo; echo "== profile $a $w-p1 (one cold 16K prompt)"; pf "$RES/$a/$w-p1"; done; done
  [ -s "$RES/2x/k88-2x.txt" ] && { echo; cat "$RES/2x/k88-2x.txt"; }; } > "$RES/k88.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
{ echo "k88-prefill-chunks result: \`max_num_batched_tokens\` $MNBT (16K in 3 steps instead of 4 under the 3,024-token mamba alignment) against the shipped 8192 on 1x v2.2.0, ctl arm arm ctl on both Sparks, plus a probe of the K6144 N2560 MXFP8 GEMM that k81 found 4.7x slower at M 6,048 than at M 3,024. Measure only; nothing ships from this job."
  echo; echo "1x per Spark:$S (RULE: no cell worse than its noise band and at least one better = PASS, nothing either way = FLAT). Cold TTFT: $CJ. 2x: $P2."
  [ -s "$K/notes.md" ] && { echo; cat "$K/notes.md"; }
  echo; echo '```'; head -c 55000 "$RES/k88.txt"; echo '```'
  echo; echo "Traces stay on dgx-01 in \`$RES\`. Results: {RESULTS_URL}"; } > "$K/comment.md"
grep -qiE "$PRIV" "$K/comment.md" && { mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k88 comment.md matched the private pattern (not posted; $RES)"; exit 1; }
cp "$K/comment.md" "$RES/comment.md"
FINAL="DONE: k88 1x$S cold=$CJ -> P1 $P1; 2x $P2 ($RES/k88.txt)"
exit 0
