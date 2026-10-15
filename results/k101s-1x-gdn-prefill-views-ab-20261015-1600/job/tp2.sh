#!/usr/bin/env bash
# k101 2x stage (called by run.sh only when the 1x screen says RULE=WIN): VLLM_GDN_PREFILL_VIEWS=1 on the shipped 2x
# v2.1.0 (TP=2 over the pair) against itself. The 2x serves vLLM d21d7ade = 5dad364d + one commit that touches none
# of the six overlay files, so the same overlay builds on the 2x image (pin.sha256 is checked again in the build).
#   build   arm image from the 2x image on both Sparks ($K/overlay, fail-closed sha checks), local tag only
#   recipes ctl = registry 2x v2.1.0 as shipped; arm = ctl + arm image + own runtime cache + the knob (k99 mkarm pattern)
#   stage   arm cache k101 on both Sparks: no-clobber copies of the control cache dirs (b12x plans, triton, inductor,
#           vllm compile cache, ...)
#   bake    prebake 6 GiB KV pool (<= 3 tries) + one cold 16K prompt: the views line on both ranks; then one
#           full-pool boot; 0 b12x plans measured on rank 0
#   screen  ctl arm arm ctl ctl arm arm ctl (4 boots per side), warm only: warm-up 16K+64K, cold TTFT 16K/64K/128K x3
#           (cold.py, same prompts every boot), fresh-c8 T=0 acceptance per position, llama-benchy pp2048 + tg512
#           c1/c4/c8 x3
#   judge   report2x.py (k99 rules with the views path check): RULE2=WIN (a cold length better beyond noise, nothing
#           worse) | FLAT | LOSS | EMPTY
# Writes $RES2/k101-tp2.txt (last RULE2= line). Holds no lock of its own: the caller (run.sh) holds both GPU locks.
# Usage: called by run.sh | bash tp2.sh --dry-run
set -u
J=k101s; K=$HOME/GEN-AI/backlog/k101
source "$HOME/GEN-AI/backlog/lib.sh"
RES=${RES2:-${RES:-}}; [ -z "$RES" ] || mkdir -p "$RES"
IMG2=ghcr.io/ursuciprian/spark-vllm-b12x:k94-20261013-21e0b201-d21d7ade-warm   # 2x v2.1.0
VC=$(cut -c1-8 "$K/overlay/VLLM_COMMIT" 2>/dev/null); ARM2=spark-vllm-b12x:k101-gv-${VC:-none}-d21d7ade
REV=03f4a0570496bbe741189f4a2a7d611679e85406
CRC=$HOME/.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923   # the 2x's cache (k99)
ARC=$CRC/k101; ACACHE=/cache/runtime/k101; KV_PRE=6442450944
COLD=$K/cold.py; CORPUS=$R/results/corpus-code.txt; BENCHY_SRC=$G/llama-benchy-fork; MODEL=qwen3.8-flash-next
SNAP=$HOME/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd
REGS=$HOME/.cache/sparkrun/registries; REG2=qwen38-flashnext; N2=qwen3.8-flash-next-2x-dgx-spark
ORDER="ctl:1 arm:1 arm:2 ctl:2 ctl:3 arm:3 arm:4 ctl:4"; LENS=16384,65536,131072; REPS=3; RD=$RES/recipes
VIEWS='GDN prefill views: in-order'
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
hx() { if [ "$1" = localhost ]; then bash -c "$2"; else on2 "$2"; fi; }
img() { sed -n 's/^container: *//p' "$1" | head -1 | tr -d "\"' "; }
fld() { sed -n "s/^$2: *//p" "$1" | head -1 | tr -d "\"' "; }

mkarm() { # ctl.yaml arm.yaml: own name, image, runtime cache, knob; 2 lines changed, 4 added
  python3 - "$1" "$2" "$ACACHE" "$ARM2" <<'PY'
import sys
from collections import Counter
src, dst, cache, image = sys.argv[1:5]
lines = open(src).read().split("\n")
def one(pred, what):
    i = [k for k, l in enumerate(lines) if pred(l)]
    if len(i) != 1:
        sys.exit(f"mkarm: need exactly one {what}, found {len(i)}")
    return i[0]
n, c, e = one(lambda l: l.startswith("name: "), "name line"), one(lambda l: l.startswith("container: "), "container line"), one(lambda l: l == "env:", "env: block")
if any(l.strip().startswith(("XDG_CACHE_HOME:", "VLLM_CACHE_ROOT:", "VLLM_GDN_PREFILL_VIEWS")) for l in lines):
    sys.exit("mkarm: the control already sets one of the arm's variables")
add = ["  # k101 arm: GDN prefill views, own runtime cache", f'  XDG_CACHE_HOME: "{cache}"',
       f'  VLLM_CACHE_ROOT: "{cache}/vllm"', '  VLLM_GDN_PREFILL_VIEWS: "1"']
out = lines[:]
out[n] = "name: qwen3.8-flash-next-2x-k101-gv"
out[c] = f"container: {image}"
out[e + 1:e + 1] = add
a, b = Counter(lines), Counter(out)
if sum((a - b).values()) != 2 or sum((b - a).values()) != 2 + len(add):
    sys.exit("mkarm: unexpected diff")
open(dst, "w").write("\n".join(out))
PY
}
mkrecipes() {
  local f2; mkdir -p "$RD"
  f2=$(grep -rlx --include='*.yaml' "name: $N2" "$(readlink -f "$REGS/$REG2")/recipes" 2>/dev/null | head -1)
  [ -s "$f2" ] || { echo "2x recipe not resolvable from the registry ($REG2/$N2)"; return 1; }
  cp "$f2" "$RD/ctl-2x.yaml"
  [ "$(img "$RD/ctl-2x.yaml")" = "$IMG2" ] || { echo "registry 2x image is not $IMG2"; return 1; }
  [ "$(fld "$RD/ctl-2x.yaml" model_revision)" = "$REV" ] || { echo "registry 2x revision is not $REV"; return 1; }
  sed -n 1p "$RD/ctl-2x.yaml" | grep -q '^# Release: v2.1.0' || { echo "registry 2x recipe is not v2.1.0"; return 1; }
  mkarm "$RD/ctl-2x.yaml" "$RD/arm-2x.yaml" || return 1
  sed -e 's|^    --gpu-memory-utilization {gpu_memory_utilization}$|    --gpu-memory-utilization {gpu_memory_utilization} \\\n    --kv-cache-memory-bytes '"$KV_PRE"'|' "$RD/arm-2x.yaml" > "$RD/prebake-2x.yaml"
  [ "$(diff "$RD/arm-2x.yaml" "$RD/prebake-2x.yaml" | grep -c '^>')" = 2 ] && grep -qx "    --kv-cache-memory-bytes $KV_PRE" "$RD/prebake-2x.yaml" \
    || { echo "prebake-2x.yaml edit did not apply"; return 1; }; }
checks() {
  local ok=0 f t h
  for f in "$COLD" "$BL/k101s/report2x.py" "$K/overlay/Dockerfile" "$CORPUS" "$R/scripts/depth_decode_probe.py"; do [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ -d "$BENCHY_SRC" ] && [ -d "$SNAP" ] || { echo "missing llama-benchy fork or tokenizer snapshot"; ok=1; }
  for t in sparkrun uvx python3 curl rsync; do command -v $t > /dev/null || { echo "missing $t"; ok=1; }; done
  python3 "$BL/k101s/report2x.py" selftest > /dev/null || { echo "report2x.py selftest failed"; ok=1; }
  mkrecipes || ok=1
  for h in localhost $H2; do
    hx $h "docker image inspect $IMG2 > /dev/null 2>&1" || { echo "$h: $IMG2 missing"; ok=1; }
    hx $h "test -d $CRC/b12x/compile/preparation" || { echo "$h: 2x runtime cache $CRC missing"; ok=1; }; done
  [ -s "$RD/arm-2x.yaml" ] && { echo "arm diff:"; diff "$RD/ctl-2x.yaml" "$RD/arm-2x.yaml" | sed 's/^/   /'; }
  return $ok; }
if [ "${1:-}" = --dry-run ]; then RES=$(mktemp -d); RD=$RES/recipes; checks; rc=$?; rm -rf "$RES"; echo "tp2 dry-run exit=$rc"; exit $rc; fi
[ "${GPU_LOCK_HELD:-}" = 1 ] || { echo "refused: run.sh calls this under both GPU locks" >&2; exit 2; }
OUT=$RES/k101-tp2.txt; echo "k101 2x $(TZ=Europe/Bucharest date '+%F %T %Z'): VLLM_GDN_PREFILL_VIEWS=1 on the 2x v2.1.0 ($ARM2) vs shipped" > "$OUT"
fail() { echo "$*" >> "$OUT"; echo "RULE2=FAILED" >> "$OUT"; stop_all; exit 1; }
timeout 300 sparkrun registry update $REG2 >> "$RES/tp2.log" 2>&1 < /dev/null 9>&-
checks > "$RES/checks.txt" 2>&1 || fail "checks failed ($RES/checks.txt)"
stop_all
build2() { local h=$1 o=$RES/build-$1.log
  hx $h "cd $K/overlay && timeout -k 30 1800 docker build --build-arg BASE=$IMG2 -t $ARM2 ." > "$o" 2>&1; }
build2 localhost & b1=$!; build2 $H2 & b2=$!; wait $b1; r1=$?; wait $b2; r2=$?
[ $r1 = 0 ] && [ $r2 = 0 ] || fail "build failed ($RES/build-*.log)"
for h in localhost $H2; do
  hx $h "set -e; mkdir -p $ARC; for d in b12x triton inductor vllm flashinfer cute_dsl flash_attn_cute_dsl torch_extensions numba; do
           [ -d $CRC/\$d ] && cp -an $CRC/\$d $ARC/; done; du -sh $ARC | cut -f1" > "$RES/stage-$h.txt" 2>&1 || fail "stage $h failed"; done

keep_logs2() { # dir (before stop_all), k99
  local c; c=$(n0); [ -n "$c" ] || c=$(docker ps -a --format '{{.Names}}' | grep node_0 | head -1); [ -n "$c" ] || return 0
  docker exec "$c" cat /tmp/sparkrun_serve.log > "$1/serve.log" 2>&1 || docker logs "$c" > "$1/serve.log" 2>&1
  on2 'c=$(docker ps -a --format "{{.Names}}" | grep node_1 | head -1); [ -n "$c" ] && { docker exec $c cat /tmp/sparkrun_serve.log 2>/dev/null || docker logs $c 2>&1; }' > "$1/serve-node1.log" 2>&1
  docker exec "$c" sh -c "env | grep -E '^(VLLM|B12X|XDG)' | sort" > "$1/env.txt" 2>&1; }
tboot() { # recipe dir cold_ok (k99)
  local rec=$1 d=$2 s m n cap; cap=$([ "$3" = 1 ] && echo 5400 || echo 1800)
  stop_all; cp "$rec" "$d/recipe.yaml"
  ( cd "$(dirname "$rec")" && timeout -k 60 900 sparkrun run "$(basename "$rec")" --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  while :; do
    n=$(n0)
    m=$([ -n "$n" ] && docker exec "$n" grep -m1 -oE 'Directly load AOT compilation|Dynamo bytecode transform time' /tmp/sparkrun_serve.log 2>/dev/null)
    [ "$m" = "Dynamo bytecode transform time" ] && [ "$3" != 1 ] && return 2
    [ "$(health localhost)" = 200 ] && { echo "boot $(( $(date +%s) - s ))s ${m:-no AOT line}" > "$d/boot.txt"; return 0; }
    if [ $(( $(date +%s) - s )) -gt 180 ] && { [ -z "$n" ] || docker exec "$n" grep -qE 'Worker failed with error|EngineCore failed to start|Engine core initialization failed' /tmp/sparkrun_serve.log 2>/dev/null; }; then return 1; fi
    [ $(( $(date +%s) - s )) -ge $cap ] && return 1
    sleep 10; done; }
pos() { curl -s -m 10 "localhost:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
measured() { grep -ohE '\b[0-9]+ measured' "$1" 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
views2() { grep -q "$VIEWS" "$1/serve.log" && grep -q "$VIEWS" "$1/serve-node1.log"; }
d=; ok=
for t in 1 2 3; do d=$RES/bake/prebake-try$t; mkdir -p "$d"
  if tboot "$RD/prebake-2x.yaml" "$d" 1; then
    timeout -k 30 600 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens 16384 --reps 1 --seed k101b --cskip 3600000 \
      --out "$d/bake16k.json" > "$d/bake16k.txt" 2>&1
    keep_logs2 "$d"; stop_all; ok=1; break; fi
  keep_logs2 "$d"; stop_all; done
[ -n "$ok" ] || fail "prebake: 3 tries failed ($RES/bake)"
views2 "$d" || fail "prebake: views line missing on a rank ($d)"
d=$RES/bake/arm-full; mkdir -p "$d"
tboot "$RD/arm-2x.yaml" "$d" 1 || { keep_logs2 "$d"; fail "bake full-pool boot failed"; }
keep_logs2 "$d"; stop_all
echo "bake: prebake $(cat "$RES"/bake/prebake-try*/boot.txt 2>/dev/null | tail -1); full $(cat "$d/boot.txt"); rank 0 measured $(measured "$d/serve.log")" >> "$OUT"

S=$RES/2x; mkdir -p "$S"
for item in $ORDER; do who=${item%:*}; k=${item#*:}; d=$S/$who-p$k; mkdir -p "$d"
  rec=$RD/ctl-2x.yaml; [ $who = arm ] && rec=$RD/arm-2x.yaml
  tboot "$rec" "$d" 0; rc=$?
  if [ $rc -ne 0 ]; then keep_logs2 "$d"; [ $rc -eq 2 ] && echo cold > "$d/COLD" || echo failed > "$d/FAILED"; stop_all; continue; fi
  ( timeout -k 30 900 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens 16384,65536 --reps 1 --seed k101w --cskip 3000000 --out "$d/warm.json" > "$d/warm.txt" 2>&1 )
  ( timeout -k 30 2700 python3 "$COLD" run --base http://localhost:8000 --corpus "$CORPUS" --lens $LENS --reps $REPS --seed k101 --cskip 0 --out "$d/cold.json" > "$d/cold.txt" 2>&1 )
  pos > "$d/pos-c8.before"
  ( cd "$R" && timeout -k 30 500 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label fresh-c8 --corpus "$CORPUS" \
      --concurrency 8 --repeats 2 --temperature 0 --out "$d/probe-fresh-c8.json" --depth 256 --new 2048 --offset 418916 --max-tokens 512 > "$d/probe-fresh-c8.log" 2>&1 )
  pos > "$d/pos-c8.after"
  ( cd "$R" && timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
      --tokenizer "$SNAP" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 4 8 --runs 3 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
      --save-result "$d/task.csv" > "$d/task.log" 2>&1 )
  [ "$(health localhost)" = 200 ] || echo "server gone after the pass" > "$d/DIED"
  keep_logs2 "$d"; stop_all
  [ "$(measured "$d/serve.log")" = 0 ] || { echo "rank 0 measured $(measured "$d/serve.log") b12x plans" > "$d/COLD"; }; done
python3 "$BL/k101s/report2x.py" "$S" $REPS > "$RES/judge-2x.txt" 2>&1
{ echo "screen ($ORDER):"; cat "$RES/judge-2x.txt"; } >> "$OUT"
grep -q '^RULE2=' "$RES/judge-2x.txt" || echo "RULE2=EMPTY" >> "$OUT"
exit 0
