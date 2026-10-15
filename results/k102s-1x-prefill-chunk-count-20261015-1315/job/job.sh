#!/usr/bin/env bash
# k102-prefill-chunk-count (2026-10-15, 1x #27): how many prefill steps the 1x v2.3.0 really runs per cold prompt at
# max_num_batched_tokens 8192 / 9072 / 12096, why each chunk ends where it does, what each budget costs in memory, and
# a server A/B only for a budget that removes a step at 16K.
# Change: vLLM ursuciprian/vllm exp/k102-chunk-log (off 5dad364d; $K/overlay/VLLM_COMMIT): VLLM_DEBUG_PREFILL_SPLITS=1
# logs every _mamba_block_aligned_split decision (start, requested, scheduled, the stop that ended the chunk) and the
# recurrent manager's reachable reuse positions. Off by default; scheduler only; not a compile factor. Alignment, the
# mandatory recurrent-state stops and prefix caching stay exactly as shipped: only the budget changes in the A/B.
# Known (k97, dgx-01, 16K/64K): 8192 -> 4/12 steps, 9072 -> 4/9, 12096 -> 3/7; budget 9072/12096 boots dropped min
# MemAvailable to 5.0/4.6 GiB (8192: 14.3) during the cold autotune of the new capacity. The CPU replay in the branch
# predicts 128K: 8192 -> 23 steps, 9072 -> 16, 12096 -> 13.
#   build    the k102 image on each Spark (shipped v2.3.0 image + 3 files, fail-closed sha checks), local tag only
#   tests    (dgx-01) the split replay tests + the pinned split tests in the k102 image (CPU)
#   log      both Sparks in parallel, per budget: own runtime cache k102-b<budget> (copies of the control's b12x and
#            compile cache dirs), boot with the full 14 GiB KV pool (a failed boot -> 6 GiB prebake, then full again),
#            cold 16K / 64K / 128K (one each, same prompts every budget), split log -> chunk tables; min MemAvailable
#            per phase (boot = compile + autotune peak; prompts).
#   decide   the A/B budget = the one with fewer 16K steps than 8192 on both Sparks (fewest steps, then the smaller
#            budget); none -> RULE=NOSTEP, no A/B. A boot or prompt phase under 3 GiB MemAvailable disqualifies it.
#   screen   job-local Thunderdome on both Sparks: control = the shipped 1x v2.3.0 as is, arm = the k102 image at that
#            budget with the split log on, its cache from the log phase; bake: yes; ctl arm arm ctl ctl arm arm ctl;
#            T=0 probes fresh c1/c4/c8 (MTP acceptance canary), llama-benchy pp2048 c1 + tg512 c1/c4/c8 (3 runs), hook:
#            cold TTFT 16K/64K/128K x3 + the arm's logged chunks for every cold prompt.
#   judge    RULE=WIN (no cell worse on either Spark, cold 16K better beyond noise on both) | LOSS (a cell or a cold
#            length worse beyond noise) | FLAT; the arm boots must log the log-phase chunk lists, else no verdict.
# ETA ~1 h 15 min log phase (cold autotune of 9072/12096 ~12 min each, per Spark in parallel) + ~3 h 30 min screen.
# Output: $RES/{build,tests,log-dgx01,log-dgx02,ab-01,ab-02,pooled-*}, k102.txt; comment.md here + RESULT. Nothing ships.
# Contract (backlog/runner.sh): caller holds both GPU locks (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | JOBS=k102 setsid nohup bash ~/GEN-AI/backlog/runner.sh > ~/GEN-AI/backlog/runner-k102.nohup 2>&1 < /dev/null &
# Stop: kill -TERM <run.sh pid> (see k102.log). Knobs: BUDGETS ("8192 9072 12096"), AB (1; 0 = log phase only).
# Two jobs run this file through thin run.sh wrappers (the runner reads NEEDS from run.sh):
#   k102  (NEEDS dgx02)        build, CPU tests, log phase on dgx-02 only; writes the A/B budget to $P/BUDGET
#   k102s (NEEDS dgx01 dgx02)  build on dgx-01, log phase on both Sparks for 8192 and that budget (caches kept), screen
set -u
J=${J:-k102}; P=$HOME/GEN-AI/backlog/k102; K=$HOME/GEN-AI/backlog/$J; C=$HOME/GEN-AI/backlog/k101   # cold.py / cold_judge.py live in k101
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/$J-1x-prefill-chunk-count-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # x, hn, MODEL
HOSTS=$(spark_hosts | tr '\n' ' '); NAMES=$SPARKS
REG1=qwen38-flashnext-1x
REGF=$HOME/.cache/sparkrun/registries/$REG1/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
SHIP_IMG=ghcr.io/ursuciprian/spark-vllm-b12x:k95-20261013-b45fe867-5dad364d-warm   # 1x v2.3.0
VC=$(cut -c1-8 "$P/overlay/VLLM_COMMIT" 2>/dev/null); ARM_IMG=spark-vllm-b12x:k102-cl-${VC:-none}-b45fe867
RC=.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923
SHIP_VCR=vllm-b12x-b45fe867; KV=15032385536; KV_PRE=6442450944
if [ $J = k102s ]; then BUDGETS=${BUDGETS:-8192 $(cat "$P/BUDGET" 2>/dev/null)}; AB=1; else BUDGETS=${BUDGETS:-8192 9072 12096}; AB=0; fi
LENS="16384 65536 131072"; MINMEM=6.0; MINBOOT=2.5   # prompts: pool.py post-checks need 6 GiB; boot (autotune) only has to survive
CORPUS=$R/results/corpus-code.txt; RD=$K/rec
PRIVF=$BL/k98/private.pat; PRIV=$(cat "$PRIVF" 2>/dev/null)
[ -n "$PRIV" ] || { echo "missing $PRIVF" >&2; exit 2; }

mkrec() { # budget kv out: registry v2.3.0, comments stripped, k102 image, own cache, split log on
  sed '/^#/d' "$REGF" | sed -e "s|^name: .*|name: qwen3.8-flash-next-1x-k102-b$1|" -e "s|^container: .*|container: $ARM_IMG|" \
    -e "s|^  max_num_batched_tokens: 8192\$|  max_num_batched_tokens: $1|" \
    -e "s|^  kv_cache_memory_bytes: 15032385536\$|  kv_cache_memory_bytes: $2|" \
    -e "s|^  VLLM_CACHE_ROOT: \"/cache/runtime/$SHIP_VCR\"\$|  XDG_CACHE_HOME: \"/cache/runtime/k102-b$1\"\n  VLLM_CACHE_ROOT: \"/cache/runtime/k102-b$1/vllm\"\n  VLLM_DEBUG_PREFILL_SPLITS: \"1\"|" > "$3"
  grep -qx "  max_num_batched_tokens: $1" "$3" && grep -qx "  kv_cache_memory_bytes: $2" "$3" && grep -qx "container: $ARM_IMG" "$3" \
    && grep -qx '  VLLM_DEBUG_PREFILL_SPLITS: "1"' "$3" && grep -qx "  VLLM_CACHE_ROOT: \"/cache/runtime/k102-b$1/vllm\"" "$3" \
    && [ "$(sed '/^#/d' "$REGF" | diff - "$3" | grep -c '^>')" = $(( 5 + ($1 != 8192) + ($2 != KV) )) ]; }
mktd() { # job-local thunderdome + report + pool (k101's edits)
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
    "$R/scripts/thunderdome_report.py" > "$K/td_report_k102.py"
  sed "s|^REPORT = .*|REPORT = \"$K/td_report_k102.py\"|" "$BL/k90/pool.py" > "$K/pool.py"
  ln -sf "$R/scripts/paired_decode_ab.py" "$K/paired_decode_ab.py"
  [ "$(diff "$R/scripts/thunderdome_report.py" "$K/td_report_k102.py" | grep -c '^>')" = 3 ] && python3 -m py_compile "$K/td_report_k102.py" \
    && [ "$(diff "$BL/k90/pool.py" "$K/pool.py" | grep -c '^>')" = 1 ] || { echo "report/pool edits did not apply"; return 1; }; }
preflight() { # $1 = dry
  local ok=0 h f b
  for f in "$TD" "$R/scripts/thunderdome_report.py" "$R/scripts/paired_decode_ab.py" "$BL/k90/pool.py" "$BL/k70/k61.py" \
           "$C/cold.py" "$C/cold_judge.py" "$P/splits.py" "$P/hook.sh" "$P/overlay/Dockerfile" "$CORPUS"; do
    [ -s "$f" ] || { echo "missing $f"; ok=1; }; done
  [ "$(head -1 "$REGF" 2>/dev/null)" = "# Release: v2.3.0" ] || { echo "registry 1x recipe is not v2.3.0 ($REGF)"; ok=1; }
  [ "$(sed -n 's/^container: *//p' "$REGF")" = "$SHIP_IMG" ] || { echo "registry 1x container is not $SHIP_IMG"; ok=1; }
  mkdir -p "$RD"; for b in $BUDGETS; do mkrec $b $KV "$RD/b$b.yaml" && mkrec $b $KV_PRE "$RD/b$b-prebake.yaml" || { echo "recipe edits for $b did not apply"; ok=1; }; done
  sed '/^#/d' "$REGF" > "$RD/base.yaml"
  if [ $AB = 1 ]; then mktd || ok=1
    python3 "$K/pool.py" --selftest > /dev/null || { echo "pool.py selftest failed"; ok=1; }
    head -1 "$P/STATE" 2>/dev/null | grep -q '^DONE' || { echo "k102 (dgx-02 log phase) is not DONE"; ok=1; }; fi
  python3 "$C/cold.py" selftest > /dev/null && python3 "$C/cold_judge.py" --selftest > /dev/null && python3 "$P/splits.py" selftest > /dev/null \
    || { echo "cold.py / cold_judge.py / splits.py selftest failed"; ok=1; }
  bash -n "$P/hook.sh" || { echo "hook.sh syntax"; ok=1; }
  ( cd "$P/overlay" && sha256sum -c --quiet patched.sha256 ) || { echo "overlay files are not the branch versions"; ok=1; }
  for h in $HOSTS; do
    x $h "test -d \$HOME/$RC/b12x/compile/preparation" || { echo "$(hn $h): v2.3.0 runtime cache missing"; ok=1; }
    x $h "docker image inspect $SHIP_IMG > /dev/null 2>&1" || { echo "$(hn $h): $SHIP_IMG missing"; ok=1; }
    x $h "docker image inspect $ARM_IMG > /dev/null 2>&1" || { [ "${1:-}" = dry ] && echo "note: $(hn $h): $ARM_IMG not built yet (built at run time)" || { echo "$(hn $h): $ARM_IMG missing"; ok=1; }; }; done
  for f in "$RD"/*.yaml "$P"/*.sh "$P"/*.py "$P"/tests/*.py; do grep -qiE "$PRIV" "$f" && { echo "private pattern in $f"; ok=1; }; done
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  if [ $J = k102s ] && head -1 "$P/STATE" 2>/dev/null | grep -q '^DONE' && ! grep -qE '^[0-9]+$' "$P/BUDGET" 2>/dev/null; then
    echo "k102 chose no A/B budget ($(cat "$P/BUDGET" 2>/dev/null || echo 'no BUDGET file')): the run writes SKIPPED"; echo "dry-run exit=0"; exit 0; fi
  timeout 300 sparkrun registry update $REG1 > /dev/null 2>&1 < /dev/null || echo "note: 1x registry update failed (run retries)"
  preflight dry; rc=$?
  echo "image: $ARM_IMG (vLLM $(cat "$P/overlay/VLLM_COMMIT" 2>/dev/null)); Sparks: $NAMES; budgets: $BUDGETS; A/B: $AB"
  for b in $BUDGETS; do echo "b$b recipe diff vs registry:"; sed '/^#/d' "$REGF" | diff - "$RD/b$b.yaml" | sed 's/^/   /'; done 2>/dev/null | head -40
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  sed -n 's/^# ETA /ETA /p' "$P/job.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
if [ $J = k102s ] && ! head -1 "$P/STATE" 2>/dev/null | grep -q '^DONE'; then echo "FAILED: k102s, k102 is not DONE" > "$K/STATE"; exit 0; fi
if [ $J = k102s ] && ! grep -qE '^[0-9]+$' "$P/BUDGET" 2>/dev/null; then
  echo "SKIPPED: k102s, the dgx-02 log phase found no budget that removes a 16K step ($(cat "$P/BUDGET" 2>/dev/null))" > "$K/STATE"; exit 0; fi
mkdir -p "$RES"; job_begin
echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"
mkdir -p "$RES/build" "$RES/tests" "$RES/job"
st "running: build image ($NAMES)"
build() { local h=$1 o; o=$RES/build/$(hn $1).log
  if [ $h = $H1 ]; then ( cd "$P/overlay" && timeout -k 30 1800 docker build --build-arg BASE="$SHIP_IMG" -t "$ARM_IMG" . ) > "$o" 2>&1
  else ssh -n -o ConnectTimeout=10 $H2 "mkdir -p $P" && rsync -a --delete -e "ssh -o ConnectTimeout=10" "$P/overlay" "$P/tests" "$H2:$P/" > "$o" 2>&1 \
    && ssh -n -o ConnectTimeout=10 $H2 "cd $P/overlay && timeout -k 30 1800 docker build --build-arg BASE=$SHIP_IMG -t $ARM_IMG ." >> "$o" 2>&1; fi; }
# sparkrun resolves the image on dgx-01 (the control host) even for a dgx-02 boot: build there too (docker only, no GPU)
BH=$(echo "$H1 $HOSTS" | tr ' ' '\n' | awk 'NF && !s[$0]++' | tr '\n' ' ')
BP=; for h in $BH; do build $h & BP="$BP $!"; done; CH=$BP; r=0; for p in $BP; do wait $p || r=1; done; CH=
[ $r = 0 ] || { FINAL="FAILED: $J build ($RES/build)"; exit 1; }
for h in $BH; do x $h "docker run --rm --entrypoint cat $ARM_IMG /opt/k102-overlay"; done > "$RES/build/overlay.txt" 2>&1
[ "$(sort -u "$RES/build/overlay.txt" | wc -l)" = 1 ] && grep -q "^$(cat "$P/overlay/VLLM_COMMIT") " "$RES/build/overlay.txt" \
  || { FINAL="FAILED: $J image overlay differs between Sparks or from VLLM_COMMIT ($RES/build/overlay.txt)"; exit 1; }
st "running: preflight"
timeout 300 sparkrun registry update $REG1 >> "$RES/$J.log" 2>&1 < /dev/null 9>&-
preflight > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp -r "$P"/job.sh "$P"/hook.sh "$P"/*.py "$P/overlay" "$P/tests" "$RD"/*.yaml "$RES/job/" 2>/dev/null; [ $AB = 1 ] && cp "$K"/td.sh "$RES/job/"

st "running: CPU tests (k102 image)"
TH=${HOSTS%% *}
child x $TH "docker rm -f k102-tests > /dev/null 2>&1; timeout -k 30 1800 docker run --rm --init --name k102-tests -e PYTHONDONTWRITEBYTECODE=1 \
  -v $P/tests:/k102tests:ro --entrypoint bash $ARM_IMG -c 'python3 -m pytest --version > /dev/null 2>&1 || python3 -m pip install -q pytest; cd /tmp &&
    python3 -m pytest -p no:cacheprovider --noconftest -v -rs /k102tests'" > "$RES/tests/pytest.log" 2>&1 9>&-
grep -oE '[A-Za-z0-9_]+\.py::[^ ]+ (PASSED|FAILED|ERROR|SKIPPED)' "$RES/tests/pytest.log" | sort -u > "$RES/tests/summary.txt"
grep -q 'test_mamba_prefill_split_log.py::test_reproduces_the_profiled_16k_schedules PASSED' "$RES/tests/summary.txt" \
  && ! grep -qE ' (FAILED|ERROR)$' "$RES/tests/summary.txt" || { FINAL="FAILED: k102 CPU tests ($RES/tests/pytest.log)"; exit 1; }

regard() { # host (k90)
  local h=$1 pf p; pf=$RES/.guard-$(hn $1).pid
  p=$(cat "$pf" 2>/dev/null); [ -n "$p" ] || p=$([ $h = $H1 ] && echo $G1 || echo $G2)
  kill -0 "$p" 2>/dev/null && return 0
  if [ $h = $H1 ]; then timeout 12h bash -c "$GUARD" >> "$RES/guard-dgx01.log" 2>&1 < /dev/null 9>&- &
  else timeout 12h ssh -n $H2 "$GUARD" >> "$RES/guard-dgx02.log" 2>&1 9>&- & fi
  echo $! > "$pf"; log "memory guard $(hn $h) restarted"; }
cid() { x $1 "docker ps -q --filter name=sparkrun | head -1"; }
srl() { flock -o "$RES/.sparkrun.lock" "$@"; }
down() { srl timeout -k 30 300 sparkrun stop --all --hosts $1 >> "$RES/$J.log" 2>&1 9>&-; sleep 5
  x $1 "docker ps -q --filter name=sparkrun | xargs -r docker rm -f" > /dev/null 2>&1; }
mark() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$(date +%s)" "$5" >> "$RES/marks.tsv"; }   # host budget phase start end rc
sboot() { # host dir recipe -> 0 up (k90 prebake loop, without the down)
  local h=$1 d=$2 s c f; mkdir -p "$d"; cp "$3" "$d/recipe.yaml"
  ( cd "$RD" && srl timeout -k 30 900 sparkrun run "$3" --hosts $h --solo --no-follow ) >> "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  s=$(date +%s)
  until [ "$(health $h)" = 200 ]; do
    c=$(cid $h); [ -n "$c" ] && x $h "docker exec $c cat /tmp/sparkrun_serve.log" > "$d/serve.log.tmp" 2>/dev/null && mv -f "$d/serve.log.tmp" "$d/serve.log"
    f=$(grep -cE 'Engine core initialization failed|EngineCore failed to start' "$d/serve.log" 2>/dev/null)
    if [ $(( $(date +%s) - s )) -gt 120 ] && { [ -z "$c" ] || [ "${f:-0}" -gt 0 ]; }; then echo "boot failed after $(( $(date +%s) - s )) s" > "$d/FAILED"; return 1; fi
    [ $(( $(date +%s) - s )) -gt 5400 ] && { echo "boot timeout" > "$d/FAILED"; return 1; }
    sleep 15; done
  echo "boot $(( $(date +%s) - s )) s" > "$d/boot.txt"; }
logphase() { # host: every budget in turn on this Spark
  local h=$1 n b o d a rc; n=$(hn $1)
  for b in $BUDGETS; do o=$RES/log-$n/b$b; mkdir -p "$o"; regard $h
    x $h "set -e; B=\$HOME/$RC; [ -d \"\$B/k102-b$b/vllm\" ] && { echo kept; exit 0; }; rm -rf \"\$B/k102-b$b\"; mkdir -p \"\$B/k102-b$b\"; cp -a \"\$B/b12x\" \"\$B/k102-b$b/\"; { cp -a \"\$B/$SHIP_VCR\" \"\$B/k102-b$b/vllm\" 2>/dev/null || { rm -rf \"\$B/k102-b$b/vllm\"; mkdir -p \"\$B/k102-b$b/vllm\"; }; }" \
      > "$o/seed.txt" 2>&1 || { echo "seed failed" > "$o/FAILED"; continue; }
    a=$(date +%s); sboot $h "$o/boot" "$RD/b$b.yaml"; rc=$?; mark $n $b boot $a $rc
    if [ $rc != 0 ]; then down $h; a=$(date +%s)   # k78/k88 lesson: compile + autotune with a small pool, then the full pool
      sboot $h "$o/prebake" "$RD/b$b-prebake.yaml"; rc=$?; mark $n $b prebake $a $rc; down $h
      [ $rc = 0 ] || { echo "prebake failed" > "$o/FAILED"; continue; }
      a=$(date +%s); sboot $h "$o/boot2" "$RD/b$b.yaml"; rc=$?; mark $n $b boot2 $a $rc
      [ $rc = 0 ] || { down $h; echo "full-pool boot failed after the prebake" > "$o/FAILED"; continue; }; fi
    a=$(date +%s)
    timeout -k 30 2400 python3 "$C/cold.py" run --base "http://$h:8000" --corpus "$CORPUS" --lens 16384,65536,131072 --reps 1 \
      --seed k102 --cskip 0 --out "$o/cold.json" > "$o/cold.txt" 2>&1; rc=$?; mark $n $b prompts $a $rc
    d=$(cid $h); x $h "docker exec $d cat /tmp/sparkrun_serve.log" > "$o/serve.log" 2>/dev/null; down $h
    python3 "$P/splits.py" table "$o/serve.log" > "$o/splits.txt" 2>&1
    { echo "plans measured: $(grep -ohE '\b[0-9]+ measured' "$o/serve.log" | awk '{s+=$1} END {print s+0}')"
      grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$o/serve.log"
      echo "Directly load AOT: $(grep -c 'Directly load AOT compilation' "$o/serve.log"); Dynamo bytecode transform: $(grep -c 'Dynamo bytecode transform' "$o/serve.log")"
      grep -m3 -E 'reachable boundaries' "$o/serve.log" | sed 's/^.*reachable/reachable/'; } > "$o/facts.txt" 2>&1; done; }
memrow() { # host budget -> "boot_min prompts_min" GiB (boot = the minimum over its boot, prebake and boot2 phases)
  local n=$1 b=$2 bm="" pm="" s e ph t
  while IFS=$'\t' read -r hh bb ph s e _; do [ "$hh" = $n ] && [ "$bb" = $b ] || continue
    case $ph in prompts) pm=$(minmem "$RES/guard-$n.log" $s $e) ;; *) t=$(minmem "$RES/guard-$n.log" $s $e); bm=$(awk -v a="${bm:-99}" -v b="$t" 'BEGIN { print (b + 0 < a + 0) ? b : a }') ;; esac
  done < "$RES/marks.tsv"; echo "${bm:--} ${pm:--}"; }

stop_all
st "running: log phase (budgets $BUDGETS, $NAMES)"
LP=; for h in $HOSTS; do logphase $h > "$RES/logphase-$(hn $h).log" 2>&1 & LP="$LP $!"; done
CH=$LP; for q in $LP; do wait $q; done; CH=
for h in $HOSTS; do regard $h; done
{ echo "== chunks per cold prompt (split log) and min MemAvailable (GiB: boot incl. compile/autotune, prompts; guard aborts < 2)"
  for n in $NAMES; do for b in $BUDGETS; do o=$RES/log-$n/b$b
    echo "-- $n budget $b: $(cat "$o/FAILED" 2>/dev/null)mem $(memrow $n $b); $(tr '\n' ' ' < "$o/facts.txt" 2>/dev/null)"
    cat "$o/splits.txt" 2>/dev/null; done; done; } > "$RES/logphase.txt"
# steps per budget and length on this job's Sparks; the A/B budget removes a 16K step on each and kept memory headroom
declare -A ST; BEST=; BS=99; N1=${NAMES%% *}
for b in $BUDGETS; do for n in $NAMES; do for L in $LENS; do
  s=$(python3 "$P/splits.py" steps "$RES/log-$n/b$b/serve.log" $L 2>/dev/null); ST[$n.$b.$L]=$( [ -n "$s" ] && echo "$s" | tr ',' '\n' | wc -l || echo -); done; done; done
{ echo "== steps per cold prompt ($NAMES)"; for b in $BUDGETS; do printf '  %6s:' $b
  for L in $LENS; do printf ' %3sK' $((L / 1024)); for n in $NAMES; do printf ' %s' "${ST[$n.$b.$L]}"; done; done; echo; done; } >> "$RES/logphase.txt"
INC=; for b in $BUDGETS; do for n in $NAMES; do for L in $LENS; do [ "${ST[$n.$b.$L]}" = - ] && INC="$INC $n/b$b/$((L / 1024))K"; done
  set -- $(memrow $n $b); { [ "$1" = - ] || [ "$2" = - ]; } && INC="$INC $n/b$b/mem"; done; done
[ -z "$INC" ] || { echo "log phase incomplete:$INC" >> "$RES/logphase.txt"; FINAL="FAILED: $J log phase incomplete:$INC ($RES/logphase.txt)"; exit 1; }
for b in $BUDGETS; do [ $b = 8192 ] && continue
  ok=1; for n in $NAMES; do s=${ST[$n.$b.16384]}; r=${ST[$n.8192.16384]}
    [ "$s" != - ] && [ "$r" != - ] && [ "$s" -lt "$r" ] || ok=0
    set -- $(memrow $n $b); for m in "$1:$MINBOOT" "$2:$MINMEM"; do [ "${m%%:*}" != - ] && awk -v m="${m%%:*}" -v t="${m#*:}" 'BEGIN { exit !(m >= t) }' || ok=0; done; done
  [ $ok = 1 ] && [ "${ST[$N1.$b.16384]}" -lt $BS ] && { BEST=$b; BS=${ST[$N1.$b.16384]}; }; done
echo "A/B budget: ${BEST:-none} (16K steps ${BS/99/-} vs ${ST[$N1.8192.16384]} at 8192)" >> "$RES/logphase.txt"
[ $J = k102 ] && echo "${BEST:-none}" > "$P/BUDGET"
log "$(tail -1 "$RES/logphase.txt")"

# per Spark: KILL (acceptance) or a worse cell = LOSS evidence; a FAIL from boot post-checks, a missing/failed/cold
# boot or fewer than 4 boots per side = no verdict (pool.py turns boot problems into RULE=FAIL as well)
prule() { # node dir, pooled txt -> PASS|FLAT|FAIL(acceptance)|FAIL(worse)|EMPTY(...)
  local n=$1 f=$2 r w b k=0
  for b in ctl-p1 ctl-p2 ctl-p3 ctl-p4 arm-p1 arm-p2 arm-p3 arm-p4; do
    [ -d "$n/$b" ] && [ ! -e "$n/$b/FAILED" ] && [ ! -e "$n/$b/COLD" ] && [ ! -e "$n/$b/DIED" ] && k=$((k + 1)); done
  grep -q '^VERDICT=KILL' "$f" 2>/dev/null && { echo "FAIL(acceptance)"; return; }
  grep -q '^VERDICT=' "$f" 2>/dev/null || { echo "EMPTY(no-report)"; return; }
  r=$(sed -n 's/^RULE=//p' "$f" | head -1); w=$(sed -n 's/^worse: //p' "$f" | head -1)
  [ "$r" = FAIL ] && [ -n "$w" ] && [ "$w" != none ] && { echo "FAIL(worse)"; return; }
  [ $k = 8 ] || { echo "EMPTY($k/8 boots)"; return; }
  case "$r" in PASS|FLAT) echo "$r" ;; *) echo "EMPTY(rule ${r:-none}, post checks)" ;; esac; }
# cold verdict usable only when both nodes have every length on both sides
coldok() { [ -n "$CJ" ] && ! grep -q 'missing boots' "$RES/cold-judge.txt" && [ "$(grep -cE '[0-9]+K: control' "$RES/cold-judge.txt")" -ge 6 ]; }
RU=NOSTEP; S=; CJ=; CK=; [ $AB = 0 ] && RU=LOGGED
if [ -n "$BEST" ] && [ "$AB" = 1 ]; then
  st "running: screen budget $BEST (bake + ctl arm arm ctl ctl arm arm ctl on both Sparks)"
  for n in 01 02; do a=$K/ab-$n; mkdir -p "$a"
    { echo "# k102 arm on dgx-$n: 1x v2.3.0 at max_num_batched_tokens $BEST (k102 image, split log on, cache k102-b$BEST)"
      echo "name: ab-$n"; echo "recipe: $RD/b$BEST.yaml"; echo "base: $RD/base.yaml"; echo "bake: yes"; echo "hook: $P/hook.sh"; } > "$a/thunderdome.arm"; done
  child env RES="$RES" GPU_LOCK_HELD=1 GATE=0 CHAIN_NO_RESTORE=1 timeout -k 600 36000 bash "$K/td.sh" "$K/ab-01" "$K/ab-02" > "$RES/thunderdome.nohup" 2>&1 < /dev/null; rc=$?
  V=$(head -1 "$RES/STATE" 2>/dev/null); stop_all; for h in $HOSTS; do regard $h; done
  st "running: judge"
  for a in ab-01 ab-02; do n=dgx${a#ab-}
    [ -d "$RES/$a" ] || { S="$S $a=not-run"; continue; }
    python3 "$K/pool.py" "$RES/none" "$RES/$a" "$RES/pooled-$a" > "$RES/pooled-$a.txt" 2>&1
    r=$(prule "$RES/$a" "$RES/pooled-$a.txt"); S="$S $a=$r"
    for p in "$RES/$a"/arm-p[0-9]; do for L in $LENS; do   # the arm must run the log-phase schedule on every boot
      w=$(sed -n "s/^chunks$L //p" "$p/hook/facts.txt" 2>/dev/null); e=$(python3 "$P/splits.py" steps "$RES/log-$n/b$BEST/serve.log" $L)
      [ -n "$w" ] && [ "$w" = "$e" ] || CK="$CK $a/$(basename "$p")/$((L / 1024))K:'$w'!='$e'"; done; done; done
  python3 "$C/cold.py" report "$RES/ab-01" "$RES/ab-02" > "$RES/cold-report.txt" 2>&1
  python3 "$C/cold_judge.py" "$RES" ab-01 ab-02 > "$RES/cold-judge.txt" 2>&1
  CJ=$(sed -n 's/^COLD_ALL=//p' "$RES/cold-judge.txt")
  echo "budget $BEST: thunderdome exit $rc ($V); per Spark:$S; cold $CJ; chunk check:${CK:- ok}" > "$RES/screens.txt"
  case "$S" in *=FAIL*) RU=LOSS ;;
    *not-run*|*EMPTY*|*"= "*|*=) RU=EMPTY ;;
    *) if [ -n "$CK" ] || ! coldok; then RU=EMPTY
       elif grep -q 'WORSE' "$RES/cold-judge.txt"; then RU=LOSS
       elif [ "$CJ" = PASS ]; then RU=WIN; else RU=FLAT; fi ;; esac
fi

{ echo "$J $(TZ=Europe/Bucharest date '+%F %T %Z'): prefill chunk count on 1x v2.3.0 ($ARM_IMG, vLLM $(cat "$P/overlay/VLLM_COMMIT")), budgets $BUDGETS"
  echo "RULE=$RU"; echo; cat "$RES/logphase.txt"; echo; cat "$RES/screens.txt" 2>/dev/null
  echo; echo "== CPU tests (k102 image)"; cat "$RES/tests/summary.txt"
  for a in ab-01 ab-02; do [ -d "$RES/$a" ] || continue; echo; echo "-- $a ($([ ${a#ab-} = 01 ] && echo dgx-01 || echo dgx-02))"
    grep -vE '^(  self|  cross)' "$RES/pooled-$a.txt" 2>/dev/null | sed -n '/^== acceptance/,$p'
    for b in "$RES/$a"/*-p[0-9]; do [ -d "$b" ] && echo "  $(basename "$b") hook: $(tr '\n' ' ' < "$b/hook/facts.txt" 2>/dev/null)"; done; done
  [ -s "$RES/cold-judge.txt" ] && { echo; echo "== cold TTFT"; cat "$RES/cold-judge.txt" "$RES/cold-report.txt"; }
  grep -h ABORT "$RES"/guard-dgx0?.log 2>/dev/null | sed 's/^/memory guard: /'; } > "$RES/k102.txt"
mkdir -p "$K/private"; grep -rIliE "$PRIV" "$RES" 2>/dev/null | while read -r f; do log "private pattern in $f: moved to $K/private"; mv "$f" "$K/private/$(echo "${f#$RES/}" | tr / _)"; done
SUM=$(echo "$S" | sed 's/^ //')
{ echo "$J result ($NAMES): I logged every prefill split on the 1x v2.3.0 (\`VLLM_DEBUG_PREFILL_SPLITS=1\`, vLLM \`$(cut -c1-10 "$P/overlay/VLLM_COMMIT")\`) at max_num_batched_tokens $BUDGETS, for cold 16K, 64K and 128K prompts, with the memory each budget needs$( [ $AB = 1 ] && echo ", and ran a server A/B only for a budget that removes a step at 16K"). Alignment, the mandatory recurrent-state stops and prefix caching are unchanged; only the budget moves."
  echo; echo "\`RULE=$RU\`. A/B budget: \`${BEST:-none}\`. ${SUM:+Per Spark: \`$SUM\`. Cold TTFT: \`COLD_ALL=$CJ\`.}"
  echo; case $RU in
    WIN) echo "At budget $BEST the 16K prompt runs in fewer steps, cold 16K is faster beyond boot noise on both Sparks, and no decode, probe or acceptance cell is worse." ;;
    LOSS) echo "At budget $BEST a cell or a cold length is worse beyond noise, so the larger budget does not meet my rule." ;;
    FLAT) echo "At budget $BEST nothing is worse beyond noise, but cold 16K is not faster beyond noise on both Sparks." ;;
    NOSTEP) echo "No budget removed a 16K step on both Sparks with memory headroom, so I ran no A/B." ;;
    LOGGED) echo "Log phase only (dgx-02). The budget for the A/B on both Sparks (k102s): ${BEST:-none}." ;;
    *) echo "Some measurements are missing or the arm did not run the logged schedule, so this run gives no verdict." ;; esac
  echo; echo '```'; head -c 55000 "$RES/k102.txt"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"
if grep -qiE "$PRIV" "$K/comment.md" || grep -qE '192\.168\.|dgx-0[0-9]\.local' "$K/comment.md"; then
  mv "$K/comment.md" "$K/private/"; FINAL="FAILED: $J comment.md matched the private or LAN pattern (not posted; $RES)"; exit 1; fi
cp "$K/comment.md" "$RES/comment.md"
case $RU in EMPTY) FINAL="FAILED: $J measurements missing or schedule mismatch: $SUM;${CK} ($RES/k102.txt)" ;;
  LOGGED) FINAL="DONE: $J log phase; $(grep -A4 '^== steps per cold prompt' "$RES/logphase.txt" | tail -n +2 | tr -s ' ' | tr '\n' ';') A/B budget ${BEST:-none} ($RES/k102.txt)" ;;
  *) FINAL="DONE: $J RULE=$RU budget=${BEST:-none} $SUM; cold ${CJ:-n/a} ($RES/k102.txt)" ;; esac
exit 0
