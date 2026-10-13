#!/usr/bin/env bash
# k95-1x-ship-check (2026-10-13, 1x #27): ship check of 1x v2.3.0 = v2.2.0 on the b12x large-row tile-fix image
# (k90/k90b) + --enable-prompt-tokens-details, on dgx-01. The fidelity half of the gate runs on dgx-02 (k95f).
#   image   TAG = tile-fix image + plan 33d9b32e (303 records, k90-tf on dgx-01) + AOT keys 246a576d/5af689ee/f5d573a4
#           (k90-tf), local base built by hand, CI build-b0-warm (run before this job).
#   fresh   branch recipe ($REC) under a job-local sparkrun HOME with an EMPTY runtime cache (k87 mkhome): pass = AOT
#           loaded (no Dynamo compile), 0 b12x plans measured, plan 303 records, pong, revision 03f4a057 and the flag on
#           the serve command line, cached_tokens > 0 on a repeated ~12K-token prompt.
#   gate A  on the fresh boot: stragglers c5-c16, tool-use hardmode (>= 88), TC-45 x5 (100); logits capture x2;
#           llama-benchy pp2048 c1 + tg512 c1/c8 (3 runs), compared with the k90b tile-fix arm means.
#   upgrade the same recipe with the REAL runtime cache, which holds v2.2.0's AOT artifacts under the same keys and the
#           302-record plan (what a v2.2.0 user has): boot facts, errors, plans measured, logits vs the fresh boot.
# ETA ~70 min. Contract (backlog/runner.sh): dgx01 lock; exit 75 if busy; STATE DONE/FAILED; comment.md + RESULT.
NEEDS="dgx01"
set -u
J=k95; K=$HOME/GEN-AI/backlog/k95
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k95-1x-ship-check-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"; source "$K/common.sh"
if [ "${1:-}" = --dry-run ]; then rc=0
  recok && echo "ok recipe v2.3.0" || { echo "recipe check failed"; rc=1; }
  docker image inspect $TAG >/dev/null 2>&1 && echo "ok image $TAG" || { echo "image $TAG missing"; rc=1; }
  for t in tool-eval-bench uvx python3; do command -v $t >/dev/null || { echo "missing $t"; rc=1; }; done
  [ -s "$CORPUS" ] && [ -s "$R/scripts/straggler_probe.py" ] && [ -s "$R/scripts/logits_equiv.py" ] || { echo "missing scripts/corpus"; rc=1; }
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active" >&2; exit 75; }
mkdir -p "$RES"; echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; job_begin
recok || { FINAL="FAILED: k95 recipe check"; exit 1; }
cp "$K/run.sh" "$K/common.sh" "$REC" "$RES/"
st "running: stop dgx-01"; stop_all
st "running: fresh-user boot (empty runtime cache)"
FH=$K/fresh-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M); mkhome "$FH"
F=$RES/fresh; sboot "$F" $H1 "$FH" || { FINAL="FAILED: k95 fresh boot ($F)"; exit 1; }
FW=$(facts "$F" "$FH/runtime-cache/vllm/$RCN/b12x/compile/preparation/$PLAN" $H1)
CT=$(cached $H1 2>&1 | tail -1); echo "$CT" > "$F/cached.txt"; log "fresh: $FW; $CT"
( cd "$R" && for x in a b; do timeout 300 python3 scripts/logits_equiv.py capture --base-url http://localhost:8000 --out "$F/logits-$x.json" > "$F/logits-$x.log" 2>&1; done )
st "running: gate A (stragglers, hardmode, TC-45)"
G=$F/gate; mkdir -p "$G"
( cd "$R" && child timeout 900 python3 scripts/straggler_probe.py 5 6 7 8 12 16 > "$G/straggler.log" 2>&1 )
child timeout 7200 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000 --model $MODEL > "$G/hardmode.log" 2>&1
child timeout 3600 tool-eval-bench run --hardmode --temperature 0.0 --backend vllm --timeout 600 --max-turns 32 --base-url http://localhost:8000/v1 --model $MODEL --scenarios TC-45 --trials 5 > "$G/tc45.log" 2>&1
st "running: llama-benchy"
bench fresh $H1 c1-c8 1800 --concurrency 1 8
serve_log $H1 "$F/serve-end.log"; stop_all
st "running: upgrade-path boot (real runtime cache with v2.2.0 artifacts)"
U=$RES/upgrade; mkdir -p "$U"
cp -a "$HOME/.cache/sparkrun/runtime-cache/vllm/$RCN/b12x/compile/preparation/$PLAN" "$U/plan-before.json" 2>/dev/null
if sboot "$U" $H1 ""; then UW=$(facts "$U" "$HOME/.cache/sparkrun/runtime-cache/vllm/$RCN/b12x/compile/preparation/$PLAN" $H1)
  ( cd "$R" && timeout 300 python3 scripts/logits_equiv.py capture --base-url http://localhost:8000 --out "$U/logits.json" > "$U/logits.log" 2>&1 )
  UB=ok; else UB=failed; serve_log $H1 "$U/serve.log"; facts "$U" /nonexistent $H1 > /dev/null; fi
stop_all
st "running: report"
( cd "$R"; echo "== fresh a vs fresh b (same boot)"; python3 scripts/logits_equiv.py diff "$F/logits-a.json" "$F/logits-b.json"
  echo "== fresh a vs upgrade boot"; python3 scripts/logits_equiv.py diff "$F/logits-a.json" "$U/logits.json" ) > "$RES/logits.txt" 2>&1
TD=$(grep -c ': token ' <(sed -n '/fresh a vs upgrade/,$p' "$RES/logits.txt"))
GA=$(python3 - "$G" <<'PY'
import re, sys
g = sys.argv[1]; rd = lambda f: open(f"{g}/{f}", errors="replace").read() if __import__("os").path.exists(f"{g}/{f}") else ""
bad = []
hm = re.findall(r"Quality:\s*(\d+)/100", rd("hardmode.log")); sc = re.findall(r"Score:\s*([0-9.]+)", rd("tc45.log"))
pre = re.findall(r"preemptions\s*\+?(\d+)", rd("straggler.log"))
if not hm or int(hm[-1]) < 88: bad.append(f"hardmode {hm[-1] if hm else 'missing'}")
if not sc or float(sc[-1]) < 100: bad.append(f"TC-45 {sc[-1] if sc else 'missing'}")
if len(pre) < 6 or any(int(x) for x in pre): bad.append(f"stragglers preemptions {pre}")
print(("PASS" if not bad else "FAIL (" + "; ".join(bad) + ")") + f": hardmode {hm[-1] if hm else '?'}, TC-45 {sc[-1] if sc else '?'}, straggler preemptions {'/'.join(pre)}")
PY
)
BN=$(python3 - "$RES/fresh" <<'PY'
import json, sys, glob
out = []
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    for b in d.get("benchmarks", []):
        if b.get("is_context_prefill_phase"): continue
        c = b.get("concurrency"); pp = b.get("pp_throughput", {}).get("mean"); tg = b.get("tg_throughput", {}).get("mean")
        if c == 1 and pp: out.append(f"pp2048 c1 {pp:.0f}")
        if tg: out.append(f"tg512 c{c} {tg:.1f}")
print(", ".join(out) or "no benchy result")
PY
)
CTN=$(sed -n 's/.*cached_tokens=\([0-9]*\).*/\1/p' "$F/cached.txt")
{ echo "k95 $(TZ=Europe/Bucharest date '+%F %T %Z'): 1x v2.3.0 ship check on dgx-01, image $TAG"
  echo "== fresh-user boot (empty runtime cache): $FW"; cat "$F/facts.txt"; echo "usage: $(cat "$F/cached.txt")"
  echo "== gate A: $GA"; grep -hE "Quality:|Score:" "$G/hardmode.log" "$G/tc45.log" | head -3; tail -6 "$G/straggler.log"
  echo "== llama-benchy (fresh boot, 3 runs): $BN (k90b tile-fix arm means: pp2048 c1 1773-1789, tg512 c1 57.5-57.9, c8 135.4-136.2)"
  echo "== upgrade-path boot (v2.2.0 runtime cache): boot $UB ${UW:-}"; cat "$U/facts.txt" 2>/dev/null
  echo "plan file before: $(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["records"]))' "$U/plan-before.json" 2>/dev/null) records"
  cat "$RES/logits.txt"; } > "$RES/k95.txt"
V=PASS; why=
[ "$FW" = WARM ] || { V=FAIL; why="$why fresh boot not warm;"; }
sed -n 2p "$F/boot.txt" | grep -qiE '^pong .*pong' || { V=FAIL; why="$why no pong;"; }
grep -q -- "--revision $REV" "$F/cmdline.txt" && grep -q -- '--enable-prompt-tokens-details' "$F/cmdline.txt" || { V=FAIL; why="$why cmdline;"; }
[ "${CTN:-0}" -gt 0 ] 2>/dev/null || { V=FAIL; why="$why cached_tokens ${CTN:-missing};"; }
[ "${GA%%:*}" = PASS ] || { V=FAIL; why="$why gate A ${GA%%:*};"; }
UP=OK; { [ "$UB" = ok ] && [ -s "$U/logits.json" ] && [ "$(grep -oE "Traceback: [0-9]+" "$U/facts.txt")" = "Traceback: 0" ] && ! grep -q "'plan belongs to': [1-9]" "$U/facts.txt" && [ "$TD" -le 2 ]; } || UP=CHECK
FINAL="DONE: k95 $V fresh=$FW gateA=${GA%%:*} cached_tokens=${CTN:-?} upgrade=$UP ($RES/k95.txt)${why:+ --$why}"
{ echo "k95-1x-ship-check result: I checked the 1x v2.3.0 candidate (v2.2.0 on the b12x large-row tile-fix image, plus \`--enable-prompt-tokens-details\`) on one Spark. Image \`$TAG\`."
  echo; echo "- Fresh-user boot (a new sparkrun HOME with an empty runtime cache, plain \`sparkrun run\` of the recipe): $(sed -n 1p "$F/boot.txt" | sed 's/boot_s /up in /') s, $FW: $(sed -n 4,5p "$F/facts.txt" | tr '\n' ' ')"
  echo "- Usage field: on the second of two identical prompts, $(cat "$F/cached.txt")."
  echo "- Gate part on this Spark: $GA. Long-context retrieval runs on the other Spark (k95f)."
  echo "- llama-benchy on the same boot: $BN; the tile-fix arm of k90b had pp2048 c1 1773 to 1789, tg512 c1 57.5 to 57.9 and c8 135.4 to 136.2 tok/s."
  echo "- Upgrade path (the runtime cache of a v2.2.0 user, which holds v2.2.0's torch AOT artifacts under the same keys): boot $UB, $UP; logits against the fresh boot: $(sed -n '/fresh a vs upgrade/{n;p;}' "$RES/logits.txt")"
  echo; echo "Verdict: $V.${why:+ Reasons:$why}"; echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"; cp "$K/comment.md" "$RES/comment.md"
exit 0
