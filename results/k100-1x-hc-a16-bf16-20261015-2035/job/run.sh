#!/usr/bin/env bash
# k100-1x-hc-a16-bf16 (2026-10-15, 1x #64, part of #27): faster hyperconnection (HC) down projection in prefill.
# Lead and correction: k90's 1.98x for K10240 N336 (2.70 -> 1.36 ms at M 6048) compared the quantized regime with a
# prequantized GEMM alone. k96 timed the full op as served: the HC down projection is served A16 (BF16 activations,
# plan record a16 n128 k64 s1) at 1.883 ms; the forced 128x128/BK64 quantized tile is 2.228 ms and fails numerics.
# That lead is closed. The A16 kernel only has 16-row M tiles, so at prefill rows it dequantizes the 3.4 MB weight once
# per 16 rows. Arm: b12x exp/k100-hc-a16-bf16 a1ad1b69 (on b45fe867, the 1x v2.3.0 b12x): the capacity regime of an
# MXFP8 A16 query with K >= 8192 and N <= 512 sends live M >= 1024 outside graph capture and outside preparation calls
# to an exact BF16 dequantization + one BF16 GEMM (FP32 accumulation and output, one rounding). Exact-M decode regimes,
# M < 1024, selection races and every other shape keep the A16 kernel; queries, knobs, plan population, workspace
# sizes do not change.
# Two runner jobs share this script (the Sparks are split between agents, 2026-10-15):
#   k100   (NEEDS dgx01) P0 + P1
#     P0   hc_probe.py in the shipped 1x v2.3.0 image (control = the served plan record via the image seed and the
#          host runtime cache's records; candidates deq-pc = what the patch does, deq-cached, deq-b16, deq-up; decode
#          rows M 1-40 through exact-M regimes; HC norm/mix kernels; prospective capacities 9072/12096) and the TP=2 row
#          in the 2x v2.1.0 image (HC projections are replicated: same shapes, its own seed 225bf665, 16K chunks
#          5728/5728/2864/2064). PREDICT=GO needs deq-pc numerics PASS (k96 gate, every capacity-8192 cell), faster than
#          control at every probed M >= 1024, the served record as control, and a projected cold-16K saving >= 2%
#          (k81-anchored). STOP -> RULE=FLAT (probe only).
#     P1   (GO only) the arm image $K/IMAGE (CI build-b12x-overlay of a1ad1b69 on the 1x v2.3.0 image) on dgx-01: b12x
#          commit check, GPU unit tests, the probe again through the patched dispatch with P0's plan records, xcompare:
#          every decode cell and every M < 1024 byte-identical, M >= 1024 within the k96 gate and faster than shipped.
#          Pass -> DONE "A/B next: k100ab"; fail -> FAILED.
#   k100ab (NEEDS dgx01 dgx02, backlog/k100ab/run.sh) P2, only after a k100 GO + XCHECK PASS (else SKIPPED):
#     P2   control = the registry 1x v2.3.0 recipe + --profiler-config; arm = the same on the arm image with its own
#          runtime cache k100-arm (XDG and VLLM_CACHE_ROOT; b12x dir copied from the v2.3.0 cache, prebaked with a
#          6 GiB KV pool). Job-local Thunderdome on both Sparks, ctl arm arm ctl ctl arm arm ctl (4 boots per side per
#          Spark): T=0 probes fresh c1/c4/c8 (MTP acceptance per position = numerics canary, KILL beyond 0.03),
#          logits_equiv capture (quality identity, reported), llama-benchy pp2048 c1 + tg512 c1/c4/c8 (3 runs), hook:
#          cold TTFT 16K/64K/128K x3 per boot, boot facts, pass-1 torch-profiler window.
#     judge per Spark pool.py RULE (no cell worse than its band) + cold_judge.py (16K better than control boot noise,
#          no length worse). RULE=WIN: both Sparks COLD=PASS and pool PASS/FLAT; LOSS: any pool FAIL or any cold
#          length WORSE; else FLAT. 2x TP=2 A/B: a later job (k100b) on a 1x WIN.
# ETA k100: P0 ~75 min; GO adds P1 ~60 min (cold compiles in the arm image). k100ab: seed+prebake ~30, 8 boots x ~20 min.
# Output: $RES/{p0,p1,...} (k100), $RES/{rec,arm-01,arm-02,pooled-*,...} (k100ab), <job>.txt; comment.md + RESULT in
# backlog/<job>. Nothing ships from these jobs.
# Contract (backlog/runner.sh): GPU locks of NEEDS (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE
# first line DONE/SKIPPED/FAILED (lib.sh EXIT trap; restore is the runner's for single-Spark jobs).
# Usage: bash run.sh --dry-run | JOBS=k100 GRACE0=60 setsid nohup bash ~/GEN-AI/backlog/runner.sh > ... (then k100ab).
NEEDS="dgx01"
set -u
STAGE=${K100_STAGE:-probe}
[ "$STAGE" = ab ] && { NEEDS="dgx01 dgx02"; J=k100ab; } || J=k100
K=$HOME/GEN-AI/backlog/k100; KJ=$HOME/GEN-AI/backlog/$J
TAG=$([ "$STAGE" = ab ] && echo k100ab-1x-hc-a16-bf16-ab || echo k100-1x-hc-a16-bf16)
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/$TAG-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, MODEL
REG1=qwen38-flashnext-1x
REGF=$HOME/.cache/sparkrun/registries/$REG1/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
SHIP_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:k95-20261013-b45fe867-5dad364d-warm   # 1x v2.3.0
IMG2=ghcr.io/ursuciprian/spark-vllm-b12x:k94-20261013-21e0b201-d21d7ade-warm       # 2x v2.1.0 (TP=2 probe row only)
ARM_IMG=$(cat "$K/IMAGE" 2>/dev/null); ARM_SHA=a1ad1b69d491333c00c603b10d1cf46f8bf8fc22
RC=.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923   # 1x model_key (k81, k90)
ACACHE=k100-arm; KV_PRE=6442450944; SHIP_ROOT=/cache/runtime/vllm-b12x-b45fe867
HFM=$HOME/.cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE
CKPT=$HFM/snapshots/03f4a0570496bbe741189f4a2a7d611679e85406
CAPD=$BL/k97/capture; CORPUS=$R/results/corpus-code.txt; RD=$K/rec; ROWS=$K/rows
ORDER="ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4"
PYT="pytest==8.4.2 pluggy==1.6.0 iniconfig==2.1.0 pygments packaging"
PRIV=$(cat "$BL/k90/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $BL/k90/private.pat" >&2; exit 2; }

probe() { # image setup out-dir rows-dir records-dir warm(0|1) [hc_probe args]: hc_probe.py all, GPU, no network
  local img=$1 setup=$2 out=$3 rows=$4 recs=$5 warm=$6 c=() w=() rc; shift 6
  [ -d "$CAPD" ] && c=(-v "$CAPD:/cap:ro")
  [ "$warm" = 1 ] && w=(-e K100_WARM=/warm -v "$HOME/$RC/b12x/compile:/warm:ro")
  mkdir -p "$out" "$rows" "$recs"; docker rm -f k100-probe > /dev/null 2>&1
  timeout -k 30 "${K100_CTIMEOUT:-4500}" docker run --rm --init --name k100-probe --gpus all --ipc=host --network none \
    --entrypoint python3 -e CUTE_DSL_ARCH=sm_121a -e B12X_COMPILE_WORKERS=2 -e PYTHONUNBUFFERED=1 -e K100_SETUP=$setup \
    -e K100_ROWS=/rows -e K100_RECORDS=/recs -e K100_BUDGET=${K100_BUDGET:-3600} -e K100_VARIANT_TIMEOUT=${K100_VT:-900} \
    -v "$K:/k100:ro" -v "$HFM:$HFM:ro" "${w[@]}" "${c[@]}" -v "$out:/out" -v "$rows:/rows" -v "$recs:/recs" "$img" \
    /k100/hc_probe.py all /out --ckpt "$CKPT" $([ -d "$CAPD" ] && echo --cap-dir /cap) "$@" > "$out/probe.log" 2>&1 9>&-; rc=$?
  docker rm -f k100-probe > /dev/null 2>&1; return $rc; }
incheck() { # image setup: the probe's CPU check inside the image (API names, seed plan, checkpoint keys)
  timeout 300 docker run --rm --network none --entrypoint python3 -e K100_SETUP=$2 -e CUDA_VISIBLE_DEVICES= -v "$K:/k100:ro" \
    -v "$HFM:$HFM:ro" "$1" /k100/hc_probe.py check --ckpt "$CKPT" 2>&1 9>&- | grep -v 'warnings.warn\|UserWarning'; }
pytest_in() { # image cmd: pytest from pip into /tmp/pyl inside the image (the images have no pytest), then cmd
  docker run --rm --init --name k100-tests --gpus all --ipc=host --entrypoint bash -e CUTE_DSL_ARCH=sm_121a \
    -e XDG_CACHE_HOME=/tmp/cache -e B12X_COMPILE_WORKERS=2 -v "$K/tests:/t:ro" "$1" -c "
    timeout 300 python3 -m pip install -q --break-system-packages --target /tmp/pyl --no-deps $PYT || exit 3
    cd /tmp && PYTHONPATH=/tmp/pyl $2" 9>&-; }
ghcr_wait() { # the CI overlay build pushes the arm image; wait up to 90 min for it on ghcr before any pull (k100 run 1
  # pulled while CI was still running and failed on "manifest unknown")
  local t0=$(date +%s); until timeout 120 docker manifest inspect "$ARM_IMG" > /dev/null 2>&1; do
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && return 1; sleep 60; done; }
arm_commit() { docker run --rm --entrypoint cat "$ARM_IMG" /workspace/b12x-source-commit 2>/dev/null; }
fill() { # template out KEY=file...: {KEY} -> file contents
  python3 - "$@" <<'PY'
import sys
t = open(sys.argv[1]).read()
for kv in sys.argv[3:]:
    k, f = kv.split("=", 1)
    t = t.replace("{" + k + "}", open(f).read().rstrip("\n"))
open(sys.argv[2], "w").write(t)
PY
}
mkrecs() { # base.yaml (registry v2.3.0 + profiler), arm.yaml (arm image, own cache root), arm-prebake.yaml; arm dirs
  mkdir -p "$RD"
  [ "$(head -1 "$REGF" 2>/dev/null)" = "# Release: v2.3.0" ] || { echo "registry 1x recipe is not v2.3.0 ($REGF)"; return 1; }
  [ "$(sed -n 's/^container: *//p' "$REGF")" = "$SHIP_IMG" ] || { echo "registry 1x container is not $SHIP_IMG"; return 1; }
  grep -qx "  VLLM_CACHE_ROOT: \"$SHIP_ROOT\"" "$REGF" || { echo "registry 1x recipe has no VLLM_CACHE_ROOT $SHIP_ROOT"; return 1; }
  [ -n "$ARM_IMG" ] || { echo "no arm image in $K/IMAGE"; return 1; }
  sed '/^#/d' "$REGF" | sed "s|^    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\\$|    --profiler-config '{{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/runtime/prof-k100\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}}' \\\\\n    --kv-cache-memory-bytes {kv_cache_memory_bytes} \\\\|" > "$RD/base.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k100-arm|" -e "s|^container: .*|container: $ARM_IMG|" \
      -e "s|^  VLLM_CACHE_ROOT: \"$SHIP_ROOT\"\$|  XDG_CACHE_HOME: \"/cache/runtime/$ACACHE\"\n  VLLM_CACHE_ROOT: \"/cache/runtime/$ACACHE/vllm\"|" \
      "$RD/base.yaml" > "$RD/arm.yaml"
  sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k100-arm-prebake|" \
      -e "s|^  kv_cache_memory_bytes: 15032385536\$|  kv_cache_memory_bytes: $KV_PRE|" "$RD/arm.yaml" > "$RD/arm-prebake.yaml"
  [ "$(sed '/^#/d' "$REGF" | diff - "$RD/base.yaml" | grep -c '^>')" = 1 ] || { echo "base.yaml profiler edit did not apply"; return 1; }
  [ "$(diff "$RD/base.yaml" "$RD/arm.yaml" | grep -c '^>')" = 4 ] && grep -qx "container: $ARM_IMG" "$RD/arm.yaml" \
    && grep -qx "  VLLM_CACHE_ROOT: \"/cache/runtime/$ACACHE/vllm\"" "$RD/arm.yaml" || { echo "arm.yaml edits did not apply"; return 1; }
  [ "$(diff "$RD/arm.yaml" "$RD/arm-prebake.yaml" | grep -c '^>')" = 2 ] || { echo "arm-prebake.yaml edits did not apply"; return 1; }
  local n a; for n in 01 02; do a=$K/arm-$n; mkdir -p "$a"
    { echo "# k100 arm on dgx-$n: 1x v2.3.0 on the b12x A16 BF16 large-row image, own cache, prebaked; both sides with --profiler-config"
      echo "name: arm-$n"; echo "recipe: $RD/arm.yaml"; echo "base: $RD/base.yaml"; echo "bake: yes"; echo "hook: $K/hook.sh"; } > "$a/thunderdome.arm"; done; }
mktd() { # job-local thunderdome + report: repo root, ORDER 4 per side, probe cells fresh c1/c4/c8, benchy c1/c4/c8 x3
  local want=4
  grep -q '^\. "\$(cd "\$(dirname "\${BASH_SOURCE\[0\]}")" && pwd)/spark_env.sh"$' "$TD" && want=5
  sed -e '/^R=\$(cd "\$(dirname /c\R='"$R" \
      -e 's|^\. "\$(cd "\$(dirname "\${BASH_SOURCE\[0\]}")" \&\& pwd)/spark_env.sh"$|. "'"$R"'/scripts/spark_env.sh"|' \
      -e 's|^ORDER="ctl:1 arm:1 arm:2 ctl:2"$|ORDER="'"$ORDER"'"|' \
      -e '/^CELLS="fresh-c1/,/count-c8:0:8:2:0:300:20"$/c\CELLS="fresh-c1:256:1:2:104729:300:45 fresh-c4:256:4:2:314187:400:85 fresh-c8:256:8:2:418916:500:120"' \
      -e 's|--concurrency 1 8 --runs 2 |--concurrency 1 4 8 --runs 3 |' "$TD" > "$K/td.sh"
  [ "$(diff "$TD" "$K/td.sh" | grep -c '^>')" = $want ] && grep -qx "R=$R" "$K/td.sh" && grep -qx "ORDER=\"$ORDER\"" "$K/td.sh" \
    && grep -q -- '--concurrency 1 4 8 --runs 3 ' "$K/td.sh" && bash -n "$K/td.sh" || { echo "td.sh edits did not apply"; return 1; }
  sed -e 's/^PROBES = ("fresh-c1", "fresh-c4", "fresh-c8", "d16k-c4", "count-c8")$/PROBES = ("fresh-c1", "fresh-c4", "fresh-c8")/' \
      -e 's/^WALL = ("d16k-c8",)$/WALL = ()/' \
      -e 's/^BENCHY = ("pp2048 (c1)", "tg512 (c1)", "tg512 (c8)")$/BENCHY = ("pp2048 (c1)", "tg512 (c1)", "tg512 (c4)", "tg512 (c8)")/' \
      "$R/scripts/thunderdome_report.py" > "$K/td_report_k100.py"
  ln -sf "$R/scripts/paired_decode_ab.py" "$K/paired_decode_ab.py"   # the report imports it from its own dir
  [ "$(diff "$R/scripts/thunderdome_report.py" "$K/td_report_k100.py" | grep -c '^>')" = 3 ] && python3 -m py_compile "$K/td_report_k100.py" \
    || { echo "report edits did not apply"; return 1; }; }
host_checks() { # no containers, no GPU
  local ok=0 f h
  for f in "$K/hc_probe.py" "$K/pool.py" "$K/cold.py" "$K/cold_judge.py" "$K/hook.sh" "$K/comment.tmpl" \
           "$K/tests/test_blockscaled_a16_dequant.py" "$K/tests/test_blockscaled_large_m_regime.py" "$CKPT/model.safetensors.index.json"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  python3 -m py_compile "$K/hc_probe.py" || { echo "hc_probe.py does not compile"; ok=1; }
  python3 "$K/hc_probe.py" selftest > /dev/null || { echo "hc_probe.py selftest failed"; ok=1; }
  [ -d "$CAPD" ] || echo "note: no k97 capture in $CAPD, the probe uses synthetic rows"
  test -d "$HOME/$RC/b12x/compile/preparation" || { echo "dgx01: runtime cache ~/$RC/b12x missing"; ok=1; }
  if [ "$STAGE" = probe ]; then
    for f in "$SHIP_IMG" "$IMG2"; do docker image inspect "$f" > /dev/null 2>&1 || { echo "dgx01: $f missing"; ok=1; }; done
    if docker image inspect "$ARM_IMG" > /dev/null 2>&1; then
      [ "$(arm_commit)" = "$ARM_SHA" ] || { echo "arm image b12x commit is '$(arm_commit)', not $ARM_SHA"; ok=1; }
    else echo "note: arm image not on dgx-01 yet (pulled after PREDICT=GO)"; fi
  else
    for f in "$TD" "$R/scripts/thunderdome_report.py" "$R/scripts/paired_decode_ab.py" "$BL/k81/k81an.py" "$BL/k70/k61.py" "$CORPUS"; do
      [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
    python3 "$K/pool.py" --selftest > /dev/null || { echo "pool.py selftest failed"; ok=1; }
    python3 "$K/cold.py" selftest > /dev/null || { echo "cold.py selftest failed"; ok=1; }
    python3 "$K/cold_judge.py" --selftest > /dev/null || { echo "cold_judge.py selftest failed"; ok=1; }
    bash -n "$K/hook.sh" || { echo "hook.sh syntax"; ok=1; }
    mkrecs || ok=1
    mktd || ok=1
    f=$(grep -rlx --include='*.yaml' "name: qwen3.8-flash-next-2x-dgx-spark" "$(readlink -f "$HOME/.cache/sparkrun/registries/qwen38-flashnext")/recipes" 2>/dev/null | head -1)
    [ "$(sed -n 's/^container: *//p' "$f" 2>/dev/null | sed 's|.*:||')" = "$SHIPPED_TAG_DEFAULT" ] \
      || { echo "registry 2x container is not lib.sh's SHIPPED_TAG_DEFAULT $SHIPPED_TAG_DEFAULT (restore check would fail)"; ok=1; }
    for h in $H1 $H2; do x $h "docker image inspect $SHIP_IMG > /dev/null 2>&1" || { echo "$(hn $h): $SHIP_IMG missing"; ok=1; }; done
  fi
  for f in "$K"/*.sh "$K"/*.py "$K"/tests/*.py "$K/comment.tmpl"; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }
probe_res() { cat "$BL/k100/RESULT" 2>/dev/null; }
ab_ready() { # the k100 run this A/B builds on: PREDICT=GO and XCHECK=PASS
  local p; p=$(probe_res)
  [ -n "$p" ] && grep -q '^PREDICT=GO' "$p/p0/report-1x.txt" 2>/dev/null && grep -qx 'XCHECK=PASS' "$p/p1/xcompare.txt" 2>/dev/null; }

if [ "${1:-}" = --dry-run ]; then
  timeout 300 sparkrun registry update $REG1 > /dev/null 2>&1 < /dev/null || echo "note: 1x registry update failed (run retries)"
  host_checks; rc=$?
  echo "stage $STAGE, job $J, Sparks $NEEDS, arm image ${ARM_IMG:-none}"
  if [ "$STAGE" = ab ]; then
    ab_ready && echo "k100 probe run $(probe_res): GO + XCHECK PASS" || echo "note: no k100 GO + XCHECK PASS run yet: the job will end SKIPPED"
    if [ $rc = 0 ] && x $H1 "docker image inspect $ARM_IMG > /dev/null 2>&1" && x $H2 "docker image inspect $ARM_IMG > /dev/null 2>&1"; then
      D=$(mktemp -d); env RES="$D" GATE=0 CHAIN_NO_RESTORE=1 bash "$K/td.sh" "$K/arm-01" "$K/arm-02" --dry-run > "$K/td-dry-run.txt" 2>&1; t=$?; rm -rf "$D"
      grep -q REFUSED "$K/td-dry-run.txt" && { echo "td.sh refused an arm ($K/td-dry-run.txt)"; rc=1; }
      [ $t = 0 ] || { echo "td.sh dry-run exit $t ($K/td-dry-run.txt)"; rc=1; }
    else echo "note: thunderdome dry-run skipped until the arm image is on both Sparks (pulled by the job)"; fi; fi
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
eval "$(trap -p EXIT | sed "s/^trap -- '/trap -- 'docker rm -f k100-probe k100-tests > \/dev\/null 2>\&1; /")"   # probe containers go on any exit
mkdir -p "$KJ"; echo "$RES" > "$KJ/RESULT"; rm -f "$KJ/comment.md"

report() { # $J.txt, comment.md, FINAL from $RULE $WHY [$PRES = the probe run]
  local P=${PRES:-$RES} t
  mkdir -p "$KJ/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $KJ/private"; mv "$f" "$KJ/private/$(echo "${f#$RES/}" | tr / _)"; done
  { echo "$J $(TZ=Europe/Bucharest date '+%F %T %Z'): b12x A16 BF16 large-row path for the HC down projection (arm image ${ARM_IMG:-none}, b12x ${ARM_SHA:0:8}) vs 1x v2.3.0"
    echo "RULE=$RULE ($WHY)"
    [ "$P" != "$RES" ] && { echo; echo "== probe run (k100): $P"; sed -n '/^== projection/,/^VERDICT/p' "$P/p0/report-1x.txt" 2>/dev/null; }
    [ -s "$RES/p0/report-1x.txt" ] && { echo; echo "== P0 probe, 1x v2.3.0 image (served plan seed 33d9b32e)"; cat "$RES/p0/report-1x.txt"; }
    [ -s "$RES/p0/report-2x.txt" ] && { echo; echo "== P0 probe, 2x v2.1.0 image (TP=2 row: replicated HC, seed 225bf665)"; cat "$RES/p0/report-2x.txt"; }
    [ -s "$RES/p1/tests.txt" ] && { echo; echo "== P1 GPU unit tests in the arm image"; tail -15 "$RES/p1/tests.txt"; }
    [ -s "$RES/p1/xcompare.txt" ] && { echo; echo "== P1 arm image vs shipped (ctl-seed and decode rows through the patched dispatch)"; cat "$RES/p1/xcompare.txt"; }
    [ -s "$RES/screens.txt" ] && { echo; cat "$RES/screens.txt"
      for a in arm-01 arm-02; do echo; echo "-- $a ($([ ${a##*-} = 01 ] && echo dgx-01 || echo dgx-02))"
        grep -vE '^(  self|  cross)' "$RES/pooled-$a.txt" 2>/dev/null | sed -n '/^== acceptance/,$p'
        for b in "$RES/$a"/*-p[0-9]; do [ -d "$b" ] && echo "  $(basename "$b") hook: $(tr '\n' ' ' < "$b/hook/facts.txt" 2>/dev/null)"; done; done
      echo; echo "== cold TTFT"; cat "$RES/cold-judge.txt" "$RES/cold-report.txt" 2>/dev/null
      for a in arm-01 arm-02; do for w in ctl arm; do [ -d "$RES/$a/$w-p1" ] || continue
        echo; echo "== profile $a $w-p1 (one cold 16K prompt)"
        t=$(find "$RES/$a/$w-p1/hook" -path '*prof-k100*' -name '*.json*' 2>/dev/null | sort | head -1)
        [ -n "$t" ] && python3 "$BL/k81/k81an.py" trace "$t" 2>&1 | sed -n '1,22p' || echo "  no trace"; done; done
      echo; echo "== prebakes"; for d in "$RES"/prebake-*/try?; do [ -d "$d" ] && echo "-- ${d#$RES/} $(cat "$d/FAILED" 2>/dev/null) $(tr '\n' ' ' < "$d/boot.txt" 2>/dev/null)"; done; }
    grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/$J.txt"
  echo "$RULE ($WHY)" > "$RES/rule.txt"
  fill "$K/comment.tmpl" "$KJ/comment.md" RULE="$RES/rule.txt" REPORT=<(head -c 55000 "$RES/$J.txt")
  sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' -e "s/^k100-1x-hc-a16-bf16 result:/$TAG result:/" "$KJ/comment.md"
  if grep -qiE "$PRIV" "$KJ/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$KJ/comment.md"; then
    mv "$KJ/comment.md" "$KJ/private/"; FINAL="FAILED: $J comment.md matched the private or LAN pattern (not posted; $RES)"; return 1; fi
  cp "$KJ/comment.md" "$RES/comment.md"
  FINAL="DONE: $J RULE=$RULE $WHY ($RES/$J.txt)"; }

if [ "$STAGE" = probe ]; then
  st "running: preflight"
  { host_checks; ic=0
    incheck "$SHIP_IMG" 1x > "$RES/check-1x.txt"; grep -qx 'check ok' "$RES/check-1x.txt" || { echo "in-image check 1x failed"; ic=1; }
    incheck "$IMG2" 2x > "$RES/check-2x.txt"; grep -qx 'check ok' "$RES/check-2x.txt" || echo "note: in-image check 2x failed (the TP=2 row may be empty)"
    pytest_in "$SHIP_IMG" "timeout 120 python3 -m pytest --version" > "$RES/pytest-smoke.txt" 2>&1 || { echo "pytest does not install/run in the image ($RES/pytest-smoke.txt)"; ic=1; }
    [ $ic = 0 ]; } > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
  mkdir -p "$RES/job"; cp -r "$K"/run.sh "$K"/hc_probe.py "$K"/pool.py "$K"/cold.py "$K"/cold_judge.py "$K"/hook.sh "$K/comment.tmpl" \
    "$K/IMAGE" "$K/tests" "$RES/job/"
  stop_all
  st "running: P0 probe, shipped 1x v2.3.0 image (dgx-01)"
  rm -rf "$ROWS"; probe "$SHIP_IMG" 1x "$RES/p0/1x" "$ROWS/1x" "$RES/p0/records-1x" 1; r1=$?
  log "P0 1x probe exit=$r1"
  st "running: P0 probe, shipped 2x v2.1.0 image (TP=2 row, dgx-01)"
  probe "$IMG2" 2x "$RES/p0/2x" "$ROWS/2x" "$RES/p0/records-2x" 0 --only ctl-seed,deq-pc,dec; r2=$?
  log "P0 2x probe exit=$r2"
  K100_SETUP=1x python3 "$K/hc_probe.py" report "$RES/p0/1x" > "$RES/p0/report-1x.txt" 2>&1
  K100_SETUP=2x python3 "$K/hc_probe.py" report "$RES/p0/2x" > "$RES/p0/report-2x.txt" 2>&1
  PRED=$(sed -n 's/^PREDICT=//p' "$RES/p0/report-1x.txt"); PRED2=$(sed -n 's/^PREDICT=//p' "$RES/p0/report-2x.txt")
  [ $r1 = 0 ] && grep -q '^VERDICT ' "$RES/p0/report-1x.txt" || { FINAL="FAILED: k100 P0 1x probe exit $r1 ($RES/p0/1x/probe.log)"; exit 1; }
  log "P0 1x PREDICT=$PRED; 2x PREDICT=${PRED2:-none} (exit $r2)"
  if [ "${PRED%% *}" != GO ]; then
    RULE=FLAT; WHY="probe only, no A/B: PREDICT=$PRED; 2x row PREDICT=${PRED2:-none}"; report; exit 0; fi

  st "running: P1 arm image, GPU unit tests, arm probe (dgx-01)"
  mkdir -p "$RES/p1"
  docker image inspect "$ARM_IMG" > /dev/null 2>&1 || { ghcr_wait && child timeout 3600 docker pull -q "$ARM_IMG" >> "$RES/p1/pull.txt" 2>&1 9>&-; } \
    || { FINAL="FAILED: pull $ARM_IMG on dgx01 (not on ghcr after 90 min or pull error; $RES/p1/pull.txt)"; exit 1; }
  [ "$(arm_commit)" = "$ARM_SHA" ] || { FINAL="FAILED: arm image b12x commit is '$(arm_commit)', not $ARM_SHA"; exit 1; }
  incheck "$ARM_IMG" 1x > "$RES/check-arm.txt"; grep -q '^arm image: _a16_dequant MIN_ROWS' "$RES/check-arm.txt" && grep -qx 'check ok' "$RES/check-arm.txt" \
    || { FINAL="FAILED: in-image check of the arm image ($RES/check-arm.txt)"; exit 1; }
  child pytest_in "$ARM_IMG" "timeout -k 10 2400 python3 -m pytest -q -p no:cacheprovider -rs /t/test_blockscaled_a16_dequant.py /t/test_blockscaled_large_m_regime.py" \
    > "$RES/p1/tests.txt" 2>&1; rc=$?
  docker rm -f k100-tests > /dev/null 2>&1
  [ $rc = 0 ] && ! grep -qE '[0-9]+ skipped' "$RES/p1/tests.txt" \
    || { FINAL="FAILED: k100 GPU unit tests in the arm image (exit $rc, a skip counts as a failure; $RES/p1/tests.txt)"; exit 1; }
  st "running: P1 arm probe (patched dispatch, P0 plan records) + xcompare"
  K100_BUDGET=5400 K100_VT=2400 K100_CTIMEOUT=6000 probe "$ARM_IMG" 1x "$RES/p1/arm" "$ROWS/arm" "$RES/p0/records-1x" 0 --only ctl-seed,dec; rc=$?
  timeout 600 docker run --rm --network none --entrypoint python3 -e K100_SETUP=1x -v "$K:/k100:ro" -v "$RES/p0/1x:/old:ro" \
    -v "$RES/p1/arm:/new:ro" -v "$ROWS/1x:/ro:ro" -v "$ROWS/arm:/rn:ro" "$ARM_IMG" /k100/hc_probe.py xcompare /old /new /ro /rn \
    > "$RES/p1/xcompare.txt" 2>&1 9>&-
  [ $rc = 0 ] && grep -qx 'XCHECK=PASS' "$RES/p1/xcompare.txt" \
    || { FINAL="FAILED: k100 arm vs shipped probe check (arm probe exit $rc; $RES/p1/xcompare.txt)"; exit 1; }
  RULE=GO; WHY="probe PREDICT=$PRED, arm XCHECK=PASS; serving A/B next: k100ab (both Sparks); 2x row PREDICT=${PRED2:-none}"
  report; exit 0; fi

# ---------------------------------------------------------------- STAGE ab (k100ab, both Sparks)
PRES=$(probe_res)
if ! ab_ready; then FINAL="SKIPPED: k100ab, no k100 run with PREDICT=GO and XCHECK=PASS (${PRES:-none})"
  trap - EXIT; kill $G1 $G2 2>/dev/null; st "$FINAL"; exit 0; fi
st "running: preflight"
timeout 300 sparkrun registry update $REG1 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
host_checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job"; cp -r "$K"/run.sh "$K"/pool.py "$K"/cold.py "$K"/cold_judge.py "$K"/hook.sh "$K"/td.sh "$K"/td_report_k100.py \
  "$K/comment.tmpl" "$K/IMAGE" "$RD"/*.yaml "$RES/job/"
st "running: pull arm image on both Sparks"
docker image inspect "$ARM_IMG" > /dev/null 2>&1 || { ghcr_wait && child timeout 3600 docker pull -q "$ARM_IMG" >> "$RES/pull.txt" 2>&1 9>&-; } \
  || { FINAL="FAILED: pull $ARM_IMG on dgx01 ($RES/pull.txt)"; exit 1; }
x $H2 "docker image inspect $ARM_IMG > /dev/null 2>&1" || child timeout 3600 ssh -n -o ConnectTimeout=10 $H2 "docker pull -q $ARM_IMG" \
  >> "$RES/pull.txt" 2>&1 9>&- || { FINAL="FAILED: pull $ARM_IMG on dgx02 ($RES/pull.txt)"; exit 1; }
[ "$(x $H1 "docker image inspect -f '{{.Id}}' $ARM_IMG")" = "$(x $H2 "docker image inspect -f '{{.Id}}' $ARM_IMG")" ] || { FINAL="FAILED: arm image Id differs between Sparks"; exit 1; }
[ "$(arm_commit)" = "$ARM_SHA" ] || { FINAL="FAILED: arm image b12x commit is '$(arm_commit)', not $ARM_SHA"; exit 1; }

regard() { # host (as k90)
  local h=$1 pf=$RES/.guard-$(hn $1).pid p
  p=$(cat "$pf" 2>/dev/null); [ -n "$p" ] || p=$([ $h = $H1 ] && echo $G1 || echo $G2)
  kill -0 "$p" 2>/dev/null && return 0
  if [ $h = $H1 ]; then timeout 10h bash -c "$GUARD" >> "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- &
  else timeout 10h ssh -n $H2 "$GUARD" >> "$RES/guard-dgx02.log" 2>&1 9>&- & fi
  echo $! > "$pf"; log "memory guard $(hn $h) restarted"; }
cid() { x $1 "docker ps -q --filter name=sparkrun | head -1"; }
srl() { flock -o "$RES/.sparkrun.lock" "$@"; }
down() { srl timeout -k 30 300 sparkrun stop --all --hosts $1 >> "$RES/$J.log" 2>&1 9>&-; sleep 5
  x $1 "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" > /dev/null 2>&1; }
prebake() { # host try recipe (as k90)
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
  e=$(date +%s); c=$(cid $h); x $h "docker exec $c cat /tmp/sparkrun_serve.log" > "$d/serve.log" 2>/dev/null
  { echo "boot $((e - s)) s, KV pool $KV_PRE bytes, min MemAvailable $(minmem "$RES/guard-$(hn $h).log" $s $e) GiB"
    grep -hoE '(Directly load AOT compilation|Dynamo bytecode transform time)' "$d/serve.log" | sort | uniq -c
    echo "plans measured: $(grep -ohE '\b[0-9]+ measured' "$d/serve.log" | awk '{s+=$1} END {print s+0}')"
    echo "pong: $(pong $h)"; } > "$d/boot.txt" 2>&1
  down $h; }
prebake_loop() { local t; for t in 1 2 3; do prebake $1 $t "$2" && return 0; regard $1; log "prebake $(hn $1) try $t failed"; done; return 1; }
seed() { # copy the v2.3.0 b12x dir into the arm cache on both Sparks (as k90)
  local h; for h in $H1 $H2; do
    x $h "set -e; B=\$HOME/$RC; rm -rf \"\$B/$ACACHE\"; mkdir -p \"\$B/$ACACHE\"; cp -a \"\$B/b12x\" \"\$B/$ACACHE/\"; du -sh \"\$B/$ACACHE\" | cut -f1" \
      > "$RES/seed-$(hn $h).txt" 2>&1 || return 1; done; }

stop_all
st "running: P2 seed + prebake (arm cache $ACACHE, 6 GiB KV pool, both Sparks)"
seed || { FINAL="FAILED: k100ab seed of $ACACHE ($RES/seed-*.txt)"; exit 1; }
prebake_loop $H1 "$RD/arm-prebake.yaml" > "$RES/prebake-dgx01.log" 2>&1 & q1=$!
prebake_loop $H2 "$RD/arm-prebake.yaml" > "$RES/prebake-dgx02.log" 2>&1 & q2=$!
CH="$q1 $q2"; wait $q1; p1=$?; wait $q2; p2=$?; CH=
regard $H1; regard $H2
PM=$(cat "$RES"/prebake-*/try?/boot.txt 2>/dev/null | sed -n 's/^plans measured: //p' | awk '{s+=$1} END {print s+0}')
OK=; [ $p1 = 0 ] && OK="$K/arm-01"; [ $p2 = 0 ] && OK="$OK $K/arm-02"
[ -n "$OK" ] || { FINAL="FAILED: k100ab prebake on both Sparks ($RES/prebake-*)"; exit 1; }
st "running: P2 screen (ctl arm arm ctl ctl arm arm ctl on both Sparks)"
child env RES="$RES" GPU_LOCK_HELD=1 GATE=0 CHAIN_NO_RESTORE=1 timeout -k 600 36000 bash "$K/td.sh" $OK > "$RES/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
V=$(head -1 "$RES/STATE" 2>/dev/null)
stop_all; regard $H1; regard $H2
st "running: judge"
S=; for a in arm-01 arm-02; do
  [ -d "$RES/$a" ] || { S="$S $a=not-run"; continue; }
  python3 "$K/pool.py" "$RES/none" "$RES/$a" "$RES/pooled-$a" > "$RES/pooled-$a.txt" 2>&1
  r=$(sed -n 's/^RULE=//p' "$RES/pooled-$a.txt" | head -1)
  grep -q '^VERDICT=' "$RES/pooled-$a.txt" || r="FAIL(no-report)"
  S="$S $a=$r"; done
python3 "$K/cold.py" report "$RES/arm-01" "$RES/arm-02" > "$RES/cold-report.txt" 2>&1
python3 "$K/cold_judge.py" "$RES" arm-01 arm-02 > "$RES/cold-judge.txt" 2>&1
CJ=$(sed -n 's/^COLD_ALL=//p' "$RES/cold-judge.txt")
echo "screen: thunderdome exit $rc ($V); per Spark:$S; cold $CJ" > "$RES/screens.txt"
if echo "$S" | grep -q '=FAIL' || grep -q ' WORSE$' "$RES/cold-judge.txt"; then RULE=LOSS
elif [ "$CJ" = PASS ] && echo "$S" | grep -qxE '( arm-0[12]=(PASS|FLAT)){2}'; then RULE=WIN
else RULE=FLAT; fi   # a Spark that did not run (prebake failed) can only give FLAT
WHY="per Spark:$S; cold $CJ; probe $(sed -n 's/^PREDICT=//p' "$PRES/p0/report-1x.txt")"
[ "$PM" != 0 ] && WHY="$WHY; note: arm prebakes measured $PM b12x plans (selection may differ from control)"
[ $RULE = WIN ] && WHY="$WHY; 2x: k100b (TP=2 A/B, not run by this job)"
echo "$WHY" >> "$RES/screens.txt"
report || exit 1
exit 0
