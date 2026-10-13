#!/usr/bin/env bash
# k96-hc-gemm-fair-probe (2026-10-13, 1x #27): the two hyperconnection GEMMs (down K10240 -> N336, up K320 -> N10240)
# timed as served, control against forced large-row tiles, on identical inputs and weights. k90's candidate() timed a
# prequantized dense GEMM on other random inputs (2.70 vs 1.36 ms at M 6048); here every variant goes through
# blockscaled.plan_regimes + blockscaled.mm with vLLM's query (activation quantization, provided workspace), and the
# forced tiles enter through the same dispatch (b12x _preparation patched inside the probe subprocess only).
#   P0  stop dgx-01 serving, run hc_probe.py all in the 1x v2.3.0 image on dgx-01 (container k96-probe, --init,
#       --network none, timeout 55 min, probe budget 50 min, variants in value order; ctl-seed tries the served plan record), compare in the same container,
#       remove the container. Inputs: the k97 capture (~/GEN-AI/backlog/k97/capture, real layer 5 HC activations of a
#       cold 16K prompt) when present, else fixed-seed synthetic rows (labelled in the report).
#   M 1264 / 3024 / 4536 / 6048 / 8192 at capacity 8192, 9072 and 12096 at their own capacity (with 1264 / 3024).
#   Report: full-op ms, numerics vs control (sha1, cos, max rel, error vs an FP32 reference; gate before any speed
#   claim), projected seconds saved per cold 16K prompt from the budget's chunk list x 96 calls per step, and the
#   anchor against k81's 48.7 us/token dense-HC class. Measure only; nothing ships.
# ETA ~55 min: stop ~1 + probe <= 52 + report 1 (the runner restores the 2x after the job).
# Output: $RES/p0/*.tsv, *.log, compare.tsv (sampled output rows stay in the container), k96.txt; comment.md + RESULT here.
# Contract (backlog/runner.sh): dgx01 lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE first
# line DONE/FAILED (lib.sh EXIT trap). Usage: bash run.sh --dry-run | via the runner (JOBS="k97 k96 k98"). Stop: kill -TERM <pid>.
NEEDS="dgx01"
set -u
J=k96; K=$HOME/GEN-AI/backlog/k96
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k96-hc-gemm-fair-probe-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"
IMG=ghcr.io/ursuciprian/spark-vllm-b12x:k95-20261013-b45fe867-5dad364d-warm   # 1x v2.3.0 (b12x b45fe867, vLLM 5dad364d)
HFM=$HOME/.cache/huggingface/hub/models--ursuciprian--Qwen3.8-Flash-Next-NVFP4-GDN-MSE
CKPT=$HFM/snapshots/03f4a0570496bbe741189f4a2a7d611679e85406   # 1x served revision
CAPD=$BL/k97/capture
PRIV=$(cat "$K/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

drun() { # gpu|cpu, probe args; checkpoint mounted at its host path so blob symlinks resolve; capture mounted when present
  local g=(); [ "$1" = gpu ] && g=(--gpus all); shift
  local c=(); [ -d "$CAPD" ] && c=(-v "$CAPD:/cap:ro")
  docker run --rm --init --name k96-probe "${g[@]}" --ipc=host --network none --entrypoint python3 \
    -e CUTE_DSL_ARCH=sm_121a -e XDG_CACHE_HOME=/tmp/cache -e B12X_COMPILE_WORKERS=2 -e PYTHONUNBUFFERED=1 \
    -v "$K:/k96:ro" -v "$HFM:$HFM:ro" "${c[@]}" ${OUT:+-v "$OUT:/out"} "$IMG" /k96/hc_probe.py "$@"; }
fill() { # template out KEY=file...: {KEY} -> file contents (stdlib)
  python3 - "$@" <<'PY'
import sys
t = open(sys.argv[1]).read()
for kv in sys.argv[3:]:
    k, f = kv.split("=", 1)
    t = t.replace("{" + k + "}", open(f).read().rstrip("\n"))
open(sys.argv[2], "w").write(t)
PY
}
host_checks() { # no containers: safe while other jobs run
  local ok=0 f
  for f in "$K/hc_probe.py" "$K/comment.tmpl" "$CKPT/model.safetensors.index.json"; do [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  python3 -m py_compile "$K/hc_probe.py" || { echo "hc_probe.py does not compile"; ok=1; }
  python3 "$K/hc_probe.py" selftest > /dev/null || { echo "hc_probe.py selftest failed"; ok=1; }
  docker image inspect "$IMG" > /dev/null 2>&1 || { echo "$IMG missing on dgx-01"; ok=1; }
  for f in "$K"/run.sh "$K"/*.py "$K/comment.tmpl"; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }
preflight() { # under the lock: host checks plus the in-image CPU check
  local ok=0; host_checks || ok=1
  docker rm -f k96-probe > /dev/null 2>&1   # a stale probe container from an earlier killed run
  timeout 300 env OUT= bash -c "$(declare -f drun); K=$K HFM=$HFM IMG=$IMG CAPD=$CAPD; drun cpu check --ckpt $CKPT" \
    > "$K/check.txt" 2>&1 && grep -qx 'check ok' "$K/check.txt" || { echo "in-image CPU check failed ($K/check.txt)"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  host_checks; rc=$?
  [ -d "$CAPD" ] && echo "inputs: k97 capture ($(ls "$CAPD" | grep -c '^hc'))" || echo "inputs: no k97 capture yet, synthetic rows (run k97 first)"
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: stop dgx-01 serving; hc_probe (ctl autotuned, q-def, forced tiles, q-prop, A16 sweep) in $IMG; report"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
eval "$(trap -p EXIT | sed "s/^trap -- '/trap -- 'docker rm -f k96-probe > \/dev\/null 2>\&1; /")"   # probe container goes on any exit
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job" "$RES/p0"; cp "$K"/run.sh "$K"/hc_probe.py "$K"/comment.tmpl "$K/check.txt" "$RES/job/"
st "running: P0 stop dgx-01, HC GEMM probe"
stop_all
OUT=$RES/p0
child timeout -k 30 3300 bash -c "$(declare -f drun); K=$K HFM=$HFM IMG=$IMG CAPD=$CAPD OUT=$OUT; drun gpu all /out --ckpt $CKPT $([ -d "$CAPD" ] && echo --cap-dir /cap)" \
  > "$RES/p0/probe.log" 2>&1; rc=$?
docker rm -f k96-probe > /dev/null 2>&1   # a hard timeout kills the docker CLI, not the container
log "P0 probe exit=$rc"
st "running: report"
python3 "$K/hc_probe.py" report "$RES/p0" > "$RES/k96.txt" 2>&1
V=$(sed -n 's/^VERDICT //p' "$RES/k96.txt")
fill "$K/comment.tmpl" "$K/comment.md" VERDICT=<(echo "$V (probe exit $rc)") REPORT="$RES/k96.txt"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
grep -qiE "$PRIV" "$K/comment.md" && { mkdir -p "$K/private"; mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k96 comment.md matched the private pattern (not posted; $RES)"; exit 1; }
cp "$K/comment.md" "$RES/comment.md"
[ $rc = 0 ] && grep -q '^VERDICT ' "$RES/k96.txt" && FINAL="DONE: k96 $V ($RES/k96.txt)" || FINAL="FAILED: k96 probe exit $rc ($RES/p0/probe.log)"
exit 0
