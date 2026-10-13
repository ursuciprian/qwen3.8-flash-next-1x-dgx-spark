#!/usr/bin/env bash
# k97-chunk-split-log (2026-10-13, 1x #27): which prefill chunks the 1x v2.3.0 recipe really runs, and why, at
# max_num_batched_tokens 8192 / 9072 / 12096, plus peak memory per budget. k88 found 9072 + 3024 + 3024 + 1264 at budget
# 9072 (4 steps, not 3); this decides whether a budget A/B is worth running. No A/B, no serving change.
#   mod      mod/sitecustomize.py on PYTHONPATH (from the boot's own runtime cache, image untouched) patches vLLM source
#            text at import, env-gated: K97SPLIT logs every _mamba_block_aligned_split call (start, asked tokens,
#            block-aligned end, the stops inside the chunk: next_block / retention / last_cache / tail_hash /
#            shared_prefix / reuse = the recurrent manager's reachable boundaries, and which one won); K97STEP logs the
#            prefill tokens each step schedules. Capture hooks (8192 boot only, armed for one prompt): layer 5 and 47
#            routed-MoE inputs + topk ids/weights, layer 5 HC GEMM inputs, for k96/k98.
#   per budget (dgx-01, own sparkrun HOME + runtime cache per budget, so the shipped plan file and AOT cache stay
#            untouched): boot with the full 14 GiB KV pool; if that boot fails, prebake with 6 GiB, then boot at 14 GiB
#            again (k78/k88 lesson); cold prompts 16K and 64K (cold.py seed k97, same prompts every budget);
#            serve-log K97 lines; MemAvailable from the memory guard per phase.
#   capture  8192 boot only, after its prompts: one more cold 16K prompt with <cache>/k97cap/ARM present; files go to
#            ~/GEN-AI/backlog/k97/capture (about 0.5 GB, not in the results dir).
# ETA ~60 min: 8192 warm boot ~6 + prompts ~2 + capture ~3; 9072 and 12096 cold boots (compile + b12x autotune of the
# new capacity) ~15-20 each + prompts ~2; up to ~95 min if both need the prebake fallback.
# Output: $RES/b<budget>/{boot,prebake,...}/, b<budget>/k97.log, marks.tsv, guard-dgx01.log, k97.txt; comment.md + RESULT.
# Contract (backlog/runner.sh): dgx01 lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is busy; STATE
# first line DONE/FAILED (lib.sh EXIT trap). Usage: bash run.sh --dry-run | via the runner (JOBS=k97). Stop: kill -TERM <pid>.
NEEDS="dgx01"
set -u
J=k97; K=$HOME/GEN-AI/backlog/k97
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k97-1x-chunk-split-log-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"; source "$BL/k95/common.sh"   # x serve_log; TAG RCN CORPUS mkhome sboot
REGF=$HOME/.cache/sparkrun/registries/qwen38-flashnext-1x/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
BUDGETS="8192 9072 12096"; KV=15032385536; KV_PRE=6442450944
PRIV=$(cat "$K/private.pat" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $K/private.pat" >&2; exit 2; }

mkrec() { # budget kv out [cap]: registry v2.3.0 recipe, comments stripped, budget/KV set, the k97 mod env
  sed '/^#/d' "$REGF" | sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k97-b$1|" \
    -e "s|^  max_num_batched_tokens: 8192\$|  max_num_batched_tokens: $1|" \
    -e "s|^  kv_cache_memory_bytes: 15032385536\$|  kv_cache_memory_bytes: $2|" \
    -e "s|^env:\$|env:\n  PYTHONPATH: \"/cache/runtime/k97mod\"\n  VLLM_K97_SPLIT_LOG: \"1\"${4:+\n  VLLM_K97_CAPTURE_DIR: \"/cache/runtime/k97cap\"}|" > "$3"
  grep -qx "  max_num_batched_tokens: $1" "$3" && grep -qx "  kv_cache_memory_bytes: $2" "$3" \
    && grep -qx '  VLLM_K97_SPLIT_LOG: "1"' "$3" && grep -qx '  PYTHONPATH: "/cache/runtime/k97mod"' "$3" \
    && grep -qx "container: $TAG" "$3" && { [ -z "${4:-}" ] || grep -qx '  VLLM_K97_CAPTURE_DIR: "/cache/runtime/k97cap"' "$3"; } \
    && [ "$(sed '/^#/d' "$REGF" | diff - "$3" | grep -c '^>')" = $(( 3 + $([ -n "${4:-}" ] && echo 1 || echo 0) + ($1 != 8192) + ($2 != 15032385536) )) ]; }
mkfh() { # budget -> fresh sparkrun HOME (k95 mkhome) with the mod in its runtime cache; prints the path
  local fh=$K/home-b$1 rc; rm -rf "$fh"; mkhome "$fh"; rc=$fh/runtime-cache/vllm/$RCN
  mkdir -p "$rc/k97mod" "$rc/k97cap" && cp "$K/mod/sitecustomize.py" "$K/mod/k97cap.py" "$rc/k97mod/" && echo "$fh"; }
mark() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date +%s)" "$4" >> "$RES/marks.tsv"; }
facts() { # boot dir
  local d=$1; { echo "recipe: max_num_batched_tokens $(sed -n 's/^  max_num_batched_tokens: //p' "$d/recipe.yaml"), kv_cache_memory_bytes $(sed -n 's/^  kv_cache_memory_bytes: //p' "$d/recipe.yaml")"
    echo "boot: $(tr '\n' ' ' < "$d/boot.txt" 2>/dev/null)"
    echo "Directly load AOT: $(grep -c 'Directly load AOT compilation' "$d/serve.log"); Dynamo bytecode transform: $(grep -c 'Dynamo bytecode transform' "$d/serve.log")"
    echo "b12x plans measured: $(grep -ohE '\b[0-9]+ measured' "$d/serve.log" | awk '{s+=$1} END {print s+0}')"
    grep -m1 -oE 'Model loading took [0-9.]+ GiB' "$d/serve.log"; grep -m1 -oE 'Initial free memory [0-9.]+ GiB' "$d/serve.log"
    grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$d/serve.log"
    echo "K97MOD active lines: $(grep -c 'K97MOD active' "$d/serve.log"); k97 mod errors: $(grep -c 'k97 mod:' "$d/serve.log"); Traceback: $(grep -c Traceback "$d/serve.log")"; } > "$d/facts.txt" 2>&1; }
boot() { # budget phase kv [cap] -> 0 when up (sboot: /health + pong)
  local b=$1 ph=$2 d=$RES/b$1/$2 a=$(date +%s) rc; mkdir -p "$d"
  REC=$RES/b$b/$ph.yaml; mkrec $b $3 "$REC" ${4:-} || { log "b$b $ph: recipe edit did not apply"; mark $b $ph $a 2; return 1; }
  sboot "$d" $H1 "$FH"; rc=$?; mark $b $ph $a $rc; facts "$d"; return $rc; }
prompts() { # budget
  local b=$1 d=$RES/b$1 a=$(date +%s) rc
  timeout -k 30 1200 python3 "$K/cold.py" run --base http://localhost:8000 --corpus "$CORPUS" --lens 16384,65536 --reps 1 \
    --seed k97 --cskip 0 --out "$d/cold.json" > "$d/cold.txt" 2>&1; rc=$?; mark $b prompts $a $rc
  serve_log $H1 "$d/serve-after.log"; grep -E 'K97SPLIT|K97STEP' "$d/serve-after.log" > "$d/k97.log"
  log "b$b prompts rc=$rc, $(wc -l < "$d/k97.log") K97 lines"; }
capture() { # on the running 8192 boot
  local cd=$FH/runtime-cache/vllm/$RCN/k97cap a=$(date +%s) rc
  touch "$cd/ARM"
  timeout -k 30 900 python3 "$K/cold.py" run --base http://localhost:8000 --corpus "$CORPUS" --lens 16384 --reps 1 \
    --seed k97cap --cskip 2000000 --out "$RES/b8192/capture.json" > "$RES/b8192/capture.txt" 2>&1; rc=$?
  rm -f "$cd/ARM"; sleep 5; mark 8192 capture $a $rc
  rm -rf "$K/capture"; mv "$cd" "$K/capture"; mkdir -p "$cd"
  { echo "capture prompt rc=$rc; files in $K/capture:"
    ls "$K/capture" | sed -n 's/^\([a-z0-9]*\)\.[0-9]*\.pt$/\1/p' | sort | uniq -c
    du -sh "$K/capture" | cut -f1; } > "$RES/b8192/capture-files.txt" 2>&1
  log "capture: $(tr '\n' ' ' < "$RES/b8192/capture-files.txt")"; }
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
preflight() {
  local ok=0 f t
  for f in "$K/mod/sitecustomize.py" "$K/mod/k97cap.py" "$K/k97_report.py" "$K/cold.py" "$K/comment.tmpl" "$REGF" "$CORPUS"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  sed -n 1p "$REGF" | grep -qx '# Release: v2.3.0' || { echo "registry 1x recipe is not v2.3.0"; ok=1; }
  grep -qx "container: $TAG" "$REGF" || { echo "registry 1x container is not $TAG"; ok=1; }
  for f in "$K/mod/sitecustomize.py" "$K/mod/k97cap.py" "$K/k97_report.py" "$K/cold.py"; do python3 -m py_compile "$f" || { echo "$f does not compile"; ok=1; }; done
  python3 "$K/k97_report.py" selftest > /dev/null || { echo "k97_report.py selftest failed"; ok=1; }
  t=$(mktemp -d); for b in $BUDGETS; do mkrec $b $KV "$t/b$b.yaml" $([ $b = 8192 ] && echo cap) || { echo "recipe edit failed for b$b"; ok=1; }; done
  mkrec 12096 $KV_PRE "$t/pre.yaml" || { echo "prebake recipe edit failed"; ok=1; }; rm -rf "$t"
  docker image inspect "$TAG" > /dev/null 2>&1 || { echo "$TAG missing on dgx-01"; ok=1; }
  for f in "$K"/run.sh "$K"/*.py "$K"/mod/*.py "$K/comment.tmpl"; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  preflight; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: dgx-01, budgets $BUDGETS (14 GiB KV; prebake 6 GiB fallback), cold 16K/64K, split log; capture on the 8192 boot"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
st "running: preflight"
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
mkdir -p "$RES/job/mod"; cp "$K"/run.sh "$K"/k97_report.py "$K"/cold.py "$K"/comment.tmpl "$RES/job/"; cp "$K"/mod/*.py "$RES/job/mod/"
awk '/^MemTotal/ {print int($2/1024)}' /proc/meminfo > "$RES/memtotal.txt"
st "running: stop dgx-01"; stop_all
for B in $BUDGETS; do
  FH=$(mkfh $B) || { FINAL="FAILED: k97 home for b$B"; exit 1; }
  CAP=; [ $B = 8192 ] && CAP=cap
  st "running: b$B boot (14 GiB KV)"
  boot $B boot $KV $CAP; rc=$?
  grep -q 'k97 mod:' "$RES/b$B/boot/serve.log" 2>/dev/null && { FINAL="FAILED: k97 mod error on b$B ($RES/b$B/boot/serve.log)"; exit 1; }
  if [ $rc != 0 ]; then
    stop_all; st "running: b$B prebake (6 GiB KV) after a failed 14 GiB boot"
    boot $B prebake $KV_PRE $CAP; stop_all
    st "running: b$B boot again (14 GiB KV, warm from the prebake)"
    boot $B boot2 $KV $CAP || { log "b$B: no 14 GiB boot"; stop_all; continue; }
  fi
  grep -q 'K97MOD active' "$RES/b$B"/boot*/serve.log 2>/dev/null || { FINAL="FAILED: k97 mod not loaded on b$B (PYTHONPATH not honored? $RES/b$B)"; exit 1; }
  st "running: b$B cold prompts 16K/64K"; prompts $B
  [ $B = 8192 ] && { st "running: b8192 capture prompt"; capture; }
  stop_all
done
st "running: report"
python3 "$K/k97_report.py" report "$RES" > "$RES/k97.txt" 2>&1
V=$(sed -n 's/^VERDICT //p' "$RES/k97.txt")
{ for B in $BUDGETS; do for f in "$RES"/b$B/*/facts.txt; do [ -s "$f" ] && { echo "b$B $(basename "$(dirname "$f")"):"; sed 's/^/  /' "$f"; }; done; done; } > "$RES/facts-all.txt"
[ -s "$RES/b8192/capture-files.txt" ] || echo "no capture (the 8192 boot did not come up)" > "$RES/b8192/capture-files.txt"
fill "$K/comment.tmpl" "$K/comment.md" VERDICT=<(echo "$V") REPORT="$RES/k97.txt" FACTS="$RES/facts-all.txt" CAPTURE="$RES/b8192/capture-files.txt"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
grep -qiE "$PRIV" "$K/comment.md" && { mkdir -p "$K/private"; mv "$K/comment.md" "$K/private/"; FINAL="FAILED: k97 comment.md matched the private pattern (not posted; $RES)"; exit 1; }
cp "$K/comment.md" "$RES/comment.md"
grep -q '^VERDICT ' "$RES/k97.txt" && FINAL="DONE: k97 $V ($RES/k97.txt)" || FINAL="FAILED: k97 report ($RES/k97.txt)"
cat "$RES"/b*/k97.log 2>/dev/null | grep -q K97STEP || FINAL="FAILED: k97 no budget produced K97 lines ($RES)"   # overrides DONE
exit 0
