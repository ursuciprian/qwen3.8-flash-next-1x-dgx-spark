#!/usr/bin/env bash
# k87-fresh-boot (2026-10-10, 1x #39): what a new user sees on the first boot of 1x v2.2.0, and the v2.2.0 TTFT cell at
# ISL 16K for the README. Measure only; nothing built or shipped; the real sparkrun config and caches are not touched.
#   (a) fresh   the README quick start on dgx-02 under a job-local HOME (fresh-<stamp>/): a copy of the sparkrun
#               config.yaml (ssh key, self-update, telemetry) with no registries, its own registry clone dir and an
#               EMPTY runtime cache dir (runtime_cache.dir); HF_HOME stays the real HF cache, so checkpoint and image
#               are on disk as the README's "Boot with model and image on disk" case. Commands as in the README:
#               `sparkrun registry add https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark` and
#               `sparkrun run qwen3.8-flash-next-1x-dgx-spark --hosts <spark> --solo` (plus --no-follow so the job
#               does not attach to the log). Measures: s from `sparkrun run` to /health 200, the recipe's release line,
#               TritonBundler warnings and Traceback count in the serve log, "Directly load AOT compilation",
#               the /cache/runtime mount source (must be the empty dir).
#   (b) ttft    the pinned 1x v2.2.0 recipe with the real runtime cache on both Sparks at once; llama-benchy (k76 pp
#               args) pp 16384, tg 32, depth 0, c1, 3 runs, no warm-up run, unique prompts (cold prefix); TTFT and
#               prefill tok/s per Spark and pooled. 2x not rerun: k76's 5.644 s (sd 0.012) is the shipped v2.0.0 image.
# ETA ~35 min: fresh boot <=30 min (README: ~3 min warm, +~2 min for the Triton recompile), stop 2, pair boot 5,
# benchy ~3, restore 6.
# Output: $RES/fresh/ (sparkrun logs, serve log, mounts, fresh.txt), $RES/ttft-dgx0N/ (benchy json/log, serve log),
# k87.txt; comment.md here + RESULT. The fresh HOME stays in ~/GEN-AI/backlog/k87/ (dgx-01; runtime cache on dgx-02).
# Contract (backlog/runner.sh): caller holds the gpu-lock (GPU_LOCK_HELD=1); exit 75 untouched while another job is
# queued/running; STATE first line DONE/FAILED; EXIT trap restores the shipped 2x (lib.sh).
# Usage: bash run.sh --dry-run | flock -o ~/GEN-AI/gpu-lock env GPU_LOCK_HELD=1 bash run.sh. Stop: kill -TERM <pid>.
set -u
J=k87; K=$HOME/GEN-AI/backlog/k87
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k87-fresh-boot-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"
URL=https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark; NAME=qwen3.8-flash-next-1x-dgx-spark
R1X=$K/recipe-1x-v2.2.0.yaml; RHOME=$HOME; SR=$HOME/.local/bin/sparkrun

mkhome() { # dir -> a fresh sparkrun HOME with an empty runtime cache
  local f=$1; mkdir -p "$f/.config/sparkrun" "$f/sparkrun-cache" "$f/runtime-cache"
  cp "$RHOME/.config/sparkrun/config.yaml" "$f/.config/sparkrun/config.yaml"
  printf 'cache_dir: %s\nhf_cache_dir: %s\nruntime_cache:\n  dir: %s\n' "$f/sparkrun-cache" "$RHOME/.cache/huggingface" "$f/runtime-cache" >> "$f/.config/sparkrun/config.yaml"; }
fsr() { HOME=$FH HF_HOME=$RHOME/.cache/huggingface "$SR" "$@"; }   # sparkrun as the fresh user

checks() {
  local ok=0 t h
  [ "$(sed -n 1p "$R1X")" = "# Release: v2.2.0" ] && echo "pinned 1x v2.2.0: $(sed -n 's/^container: *//p' "$R1X")" || { echo "$R1X is not 1x v2.2.0"; ok=1; }
  for h in $H1 $H2; do x $h "docker image inspect $(sed -n 's/^container: *//p' "$R1X") > /dev/null 2>&1" && echo "$(hn $h): v2.2.0 image present" || { echo "$(hn $h): v2.2.0 image MISSING"; ok=1; }; done
  FH=$K/fresh-dry; rm -rf "$FH"; mkhome "$FH"
  fsr registry add "$URL" > "$K/dry-registry.log" 2>&1 && echo "fresh HOME: registry add OK" || { echo "fresh HOME: registry add failed ($K/dry-registry.log)"; ok=1; }
  t=$(cd "$FH" && fsr show $NAME 2>&1 | grep -m1 '^Container:')
  echo "$t" | grep -q 'tp1-d3-hf-20261010-21e0b201-5dad364d-warm' && echo "fresh HOME resolves $NAME to the v2.2.0 image" || { echo "fresh HOME resolves $NAME to: '$t'"; ok=1; }
  t=$(HOME=$FH "$RHOME/.local/share/uv/tools/sparkrun/bin/python" -c "from sparkrun.core.config import SparkrunConfig as C; c=C(); print(c.runtime_cache.get('dir'), c.hf_cache_dir)" 2>&1)
  [ "$t" = "$FH/runtime-cache $RHOME/.cache/huggingface" ] && echo "fresh config: runtime cache $FH/runtime-cache (empty), HF cache $RHOME/.cache/huggingface" || { echo "fresh config not as intended: '$t'"; ok=1; }
  rm -rf "$FH"
  t=$(cd "$R" && timeout 300 uvx --from "$BENCHY_SRC" llama-benchy --version 2>/dev/null | tail -1)
  [ -n "$t" ] && echo "benchy: $t" || { echo "llama-benchy fork not runnable"; ok=1; }
  mkdir -p "$(dirname "$RES")" && [ -w "$(dirname "$RES")" ] && echo "results dir writable: $(dirname "$RES")" || { echo "results dir not writable"; ok=1; }
  return $ok; }

if [ "${1:-}" = --dry-run ]; then
  checks; rc=$?
  others_busy && echo "other GPU jobs: busy (the runner waits)" || echo "other GPU jobs: clear"
  echo "plan: (a) fresh-HOME README quick start on dgx-02, empty runtime cache; (b) pinned v2.2.0 on both Sparks, benchy pp16384 tg32 c1 x3"
  sed -n 's/^# ETA /ETA /p' "$K/run.sh"
  echo "dry-run exit=$rc"; exit $rc; fi

need_lock
others_busy && { echo "another GPU job is active: releasing the lock, nothing touched" >&2; exit 75; }
job_begin
echo "$RES" > "$K/RESULT"; : > "$K/comment.md"
st "running: preflight"
checks > "$RES/preflight.txt" 2>&1 || { FINAL="FAILED: preflight ($RES/preflight.txt)"; exit 1; }
cp "$K/run.sh" "$R1X" "$RES/"; mkdir -p "$RES/fresh"

st "running: (a) fresh-user boot on dgx-02 (README quick start, empty runtime cache)"; stop_all
FH=$K/fresh-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M); mkhome "$FH"; cp "$FH/.config/sparkrun/config.yaml" "$RES/fresh/config.yaml"
( cd "$FH" && fsr registry add "$URL" ) > "$RES/fresh/registry-add.log" 2>&1 < /dev/null 9>&-
( cd "$FH" && fsr show $NAME ) > "$RES/fresh/recipe-show.txt" 2>&1 < /dev/null 9>&-
t0=$(date +%s)
( cd "$FH" && timeout -k 60 1800 env HOME=$FH HF_HOME=$RHOME/.cache/huggingface "$SR" run $NAME --hosts $H2 --solo --no-follow ) > "$RES/fresh/sparkrun-run.log" 2>&1 < /dev/null 9>&-
t1=$(date +%s); up=
while [ $(( $(date +%s) - t0 )) -lt 1800 ]; do [ "$(health $H2)" = 200 ] && { up=$(( $(date +%s) - t0 )); break; }
  [ $(( $(date +%s) - t0 )) -gt 240 ] && [ -z "$(x $H2 "docker ps -q --filter name=sparkrun")" ] && break; sleep 5; done
serve_log $H2 "$RES/fresh/serve.log"
x $H2 "docker inspect \$(docker ps -q --filter name=sparkrun | head -1) --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'" > "$RES/fresh/mounts.txt" 2>&1
p=$(pong $H2)
{ echo "recipe: $(grep -m1 '^Container:' "$RES/fresh/recipe-show.txt")"
  echo "sparkrun run returned after $(( t1 - t0 )) s; /health 200 after ${up:-NEVER (30 min cap)} s from the start of sparkrun run; pong: $p"
  echo "runtime cache mount: $(grep -m1 '/cache/runtime' "$RES/fresh/mounts.txt")"
  echo "TritonBundler warning lines: $(grep -c 'TritonBundler' "$RES/fresh/serve.log")"
  echo "Traceback lines: $(grep -c 'Traceback' "$RES/fresh/serve.log")"
  echo "ERROR lines: $(grep -c ' ERROR ' "$RES/fresh/serve.log")"
  echo "AOT backbone: $(grep -c 'Directly load AOT compilation' "$RES/fresh/serve.log") 'Directly load AOT compilation' line(s)"
  echo "runtime cache after boot: $(x $H2 "du -sh $FH/runtime-cache 2>/dev/null | cut -f1")"
  echo "first TritonBundler lines:"; grep -m5 'TritonBundler' "$RES/fresh/serve.log" | cut -c1-240; } > "$RES/fresh/fresh.txt"
log "fresh boot: health after ${up:-never} s"
stop_all

st "running: (b) TTFT ISL 16K c1 on 1x v2.2.0, both Sparks"
pp16k() { # host
  local u=$RES/ttft-$(hn $1); mkdir -p "$u"
  ( cd "$R" && timeout -k 30 1200 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$1:8000/v1" --model $MODEL --tokenizer "$TOK" \
      --metrics-url "http://$1:8000/metrics" --save-result "$u/pp16k.json" --format json --runs 3 --prompt-mode task --no-force-length \
      --enable-prefix-caching --pp 16384 --tg 32 --depth 0 --concurrency 1 --warmup-runs 0 > "$u/pp16k.log" 2>&1 < /dev/null 9>&- )
  log "pp16k $(hn $1) exit=$?"; }
if pair ttft-dgx01 "$R1X" ttft-dgx02 "$R1X" 2400; then
  pp16k $H1 & a=$!; pp16k $H2 & b=$!; CH="$a $b"; wait $a; wait $b; CH=
else log "(b) boot failed (serve logs in ttft-*)"; fi
python3 - "$RES" > "$RES/ttft.txt" 2>&1 <<'EOF'
import json, statistics, sys
res = sys.argv[1]; pooled = {"ttft": [], "pp": []}
for h in ("dgx01", "dgx02"):
    try:
        b = [x for x in json.load(open(f"{res}/ttft-{h}/pp16k.json"))["benchmarks"] if not x.get("is_context_prefill_phase")][0]
    except Exception as ex:
        print(f"{h}: no result ({ex!r})"[:200]); continue
    t = [v / 1000 for v in b["e2e_ttft"]["values"]]; p = b["pp_throughput"]["values"]
    pooled["ttft"] += t; pooled["pp"] += p
    print(f"{h}: TTFT {statistics.mean(t):.3f} s (sd {statistics.stdev(t) if len(t) > 1 else 0:.3f}; runs {', '.join(f'{v:.2f}' for v in t)}), prefill {statistics.mean(p):.0f} tok/s")
if pooled["ttft"]:
    t = pooled["ttft"]
    print(f"pooled: TTFT {statistics.mean(t):.3f} s (sd {statistics.stdev(t) if len(t) > 1 else 0:.3f}, {len(t)} runs on 2 Sparks), prefill {statistics.mean(pooled['pp']):.0f} tok/s")
EOF
cat "$RES/fresh/fresh.txt" "$RES/ttft.txt" > "$RES/k87.txt"

st "running: report"
{ echo "Fresh-user boot of 1x v2.2.0 and the v2.2.0 TTFT cell at ISL 16K (k87-fresh-boot)."
  echo; echo "(a) I ran the README quick start on dgx-02 as a new user would: a fresh sparkrun HOME with no registries and an empty runtime cache dir (a job-local copy; my real cache was not touched), then \`sparkrun registry add $URL\` and \`sparkrun run $NAME --hosts <spark> --solo\` (with \`--no-follow\` so the job does not attach to the log). Checkpoint and image were already on disk, which is the README's \"model and image on disk\" case. Time is from the start of \`sparkrun run\` to the first \`/health\` 200."
  echo; echo '```'; cat "$RES/fresh/fresh.txt"; echo '```'
  echo; echo "(b) TTFT at ISL 16K on 1x v2.2.0 with the normal runtime cache, both Sparks at once: llama-benchy pp 16384, tg 32, depth 0, c1, 3 runs, no warm-up run, a new prompt each run (cold prefix), the same arguments as the pp sweep of k76. The 2x v2.0.0 cell is the k76 value (5.644 s, sd 0.012), measured on the shipped v2.0.0 image, so I did not rerun it."
  echo; echo '```'; cat "$RES/ttft.txt"; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
cp "$K/comment.md" "$RES/comment.md"
FINAL="DONE: k87 fresh boot health ${up:-never} s, $(grep -c TritonBundler "$RES/fresh/serve.log") TritonBundler lines; $(grep -m1 pooled "$RES/ttft.txt" | cut -c1-80)"
exit 0
