#!/usr/bin/env bash
# k95b-1x-cache-root-check (2026-10-13, 1x #27): k95 found that the v2.3.0 candidate crashes on a v2.2.0 user's runtime
# cache (same torch AOT keys, v2.2.0 artifacts loaded with the new b12x: "ValueError: plan belongs to moe.decode, not
# norm.hyperconnection"). The recipe now sets VLLM_CACHE_ROOT=/cache/runtime/vllm-b12x-b45fe867 and seeds the image's
# AOT cache there. Checks on dgx-01, same image, gate unchanged (k95 tools half + k95f fidelity half ran on the same
# model, kernels and flags; only the cache location changed):
#   fresh    empty runtime cache (job-local sparkrun HOME): AOT loaded from the new root, 0 plans measured, 303 records,
#            pong, cached_tokens > 0; logits capture
#   upgrade  the real runtime cache (v2.2.0 artifacts at the old root, 302-record plan): boots, no ValueError/Traceback
#            beyond TritonBundler, AOT from the new root, logits vs fresh
#   rollback the shipped v2.2.0 registry recipe right after: boots, pong, no ValueError (the plan file now has 303 records)
# ETA ~25 min. Contract (backlog/runner.sh): dgx01 lock; exit 75 if busy; STATE DONE/FAILED; comment.md + RESULT.
NEEDS="dgx01"
set -u
J=k95b; K=$HOME/GEN-AI/backlog/k95b
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k95b-1x-cache-root-check-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"; source "$BL/k95/common.sh"
REC=$K/recipe.yaml; ROOT=vllm-b12x-b45fe867
OLD=$HOME/.cache/sparkrun/registries/qwen38-flashnext-1x/recipes/qwen3.8-flash-next/qwen3.8-flash-next-1x-dgx-spark.yaml
recok2() { recok && grep -q "VLLM_CACHE_ROOT: \"/cache/runtime/$ROOT\"" "$REC"; }
if [ "${1:-}" = --dry-run ]; then rc=0
  recok2 && echo "ok recipe v2.3.0 with its own cache root" || { echo "recipe check failed"; rc=1; }
  sed -n 1p "$OLD" | grep -qx '# Release: v2.2.0' && echo "ok rollback recipe v2.2.0" || { echo "registry 1x recipe is not v2.2.0"; rc=1; }
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active" >&2; exit 75; }
mkdir -p "$RES"; echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; job_begin
recok2 || { FINAL="FAILED: k95b recipe check"; exit 1; }
cp "$K/run.sh" "$REC" "$RES/"
st "running: stop dgx-01"; stop_all
st "running: fresh-user boot (empty runtime cache)"
FH=$K/fresh-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M); mkhome "$FH"
F=$RES/fresh; sboot "$F" $H1 "$FH" || { FINAL="FAILED: k95b fresh boot ($F)"; exit 1; }
FW=$(facts "$F" "$FH/runtime-cache/vllm/$RCN/b12x/compile/preparation/$PLAN" $H1)
grep -q "Directly load AOT compilation from path /cache/runtime/$ROOT/" "$F/serve.log" || FW="$FW/not-from-$ROOT"
CT=$(cached $H1 2>&1 | tail -1); echo "$CT" > "$F/cached.txt"; log "fresh: $FW; $CT"
( cd "$R" && timeout 300 python3 scripts/logits_equiv.py capture --base-url http://localhost:8000 --out "$F/logits.json" > "$F/logits.log" 2>&1 )
stop_all
st "running: upgrade-path boot (real runtime cache with v2.2.0 artifacts)"
U=$RES/upgrade; RP=$HOME/.cache/sparkrun/runtime-cache/vllm/$RCN/b12x/compile/preparation/$PLAN
python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["records"]))' "$RP" > "$RES/plan-records-before.txt" 2>&1
UB=failed; if sboot "$U" $H1 ""; then UB=ok; fi
UW=$(facts "$U" "$RP" $H1)
grep -q "Directly load AOT compilation from path /cache/runtime/$ROOT/" "$U/serve.log" || UW="$UW/not-from-$ROOT"
[ $UB = ok ] && ( cd "$R" && timeout 300 python3 scripts/logits_equiv.py capture --base-url http://localhost:8000 --out "$U/logits.json" > "$U/logits.log" 2>&1 )
stop_all
st "running: rollback boot (shipped v2.2.0 recipe)"
B=$RES/rollback; mkdir -p "$B"; SREC=$REC; REC=$OLD; RB=failed; sboot "$B" $H1 "" && RB=ok; REC=$SREC
{ echo "boot: $(tr '\n' ' ' < "$B/boot.txt" 2>/dev/null)"; echo "Directly load AOT: $(grep -c 'Directly load AOT' "$B/serve.log"); Dynamo: $(grep -c 'Dynamo bytecode transform' "$B/serve.log"); ValueError: $(grep -c ValueError "$B/serve.log"); plan belongs to: $(grep -c 'plan belongs to' "$B/serve.log")"
  python3 "$R/scripts/check_seed.py" "$B/serve.log" 2>&1 | tail -1; } > "$B/facts.txt"
stop_all
( cd "$R"; echo "== fresh vs upgrade boot"; python3 scripts/logits_equiv.py diff "$F/logits.json" "$U/logits.json" ) > "$RES/logits.txt" 2>&1
TD=$(grep -c ': token ' "$RES/logits.txt")
{ echo "k95b $(TZ=Europe/Bucharest date '+%F %T %Z'): 1x v2.3.0 with VLLM_CACHE_ROOT=/cache/runtime/$ROOT, image $TAG"
  echo "== fresh (empty runtime cache): $FW"; cat "$F/facts.txt"; echo "usage: $CT"
  echo "== upgrade (v2.2.0 runtime cache, plan $(cat "$RES/plan-records-before.txt") records before): boot $UB, $UW"; cat "$U/facts.txt"
  echo "== rollback to v2.2.0 right after: boot $RB"; cat "$B/facts.txt"; cat "$RES/logits.txt"; } > "$RES/k95b.txt"
V=PASS; why=
[ "$FW" = WARM ] || { V=FAIL; why="$why fresh $FW;"; }
[ "$UB" = ok ] && ! grep -q "ValueError: [1-9]" "$U/facts.txt" && echo "$UW" | grep -q '^WARM\|^COLD' || { V=FAIL; why="$why upgrade boot $UB $UW;"; }
echo "$UW" | grep -q "not-from" && { V=FAIL; why="$why upgrade AOT not from the new root;"; }
[ "$RB" = ok ] && grep -q "ValueError: 0" "$B/facts.txt" || { V=FAIL; why="$why rollback boot $RB;"; }
[ "$TD" -le 2 ] && [ -s "$U/logits.json" ] || { V=FAIL; why="$why logits: $TD prompts with another token;"; }
CTN=$(echo "$CT" | sed -n 's/.*cached_tokens=\([0-9]*\).*/\1/p'); [ "${CTN:-0}" -gt 0 ] 2>/dev/null || { V=FAIL; why="$why cached_tokens;"; }
FINAL="DONE: k95b $V fresh=$FW upgrade=$UB/$UW rollback=$RB logits-token-diffs=$TD ($RES/k95b.txt)${why:+ --$why}"
{ echo "k95b result: the k95 upgrade-path boot of the 1x v2.3.0 candidate failed on a runtime cache from v2.2.0: both builds compute the same torch AOT keys (the b12x build is not part of the key), and the v2.2.0 graphs hold b12x plan handles that point at other plans in the new b12x (\`ValueError: plan belongs to moe.decode, not norm.hyperconnection\`). The recipe now gives v2.3.0 its own compile cache root (\`VLLM_CACHE_ROOT=/cache/runtime/$ROOT\`) and seeds the image's AOT cache there. Same image, same gate (the model, kernels and flags did not change)."
  echo; echo "- Fresh user (empty runtime cache): $(sed -n 1p "$F/boot.txt" | sed 's/boot_s /up in /') s, $FW; usage on a repeated prompt: $CT."
  echo "- Upgrade from v2.2.0 (my real runtime cache): boot $UB, $UW; $(sed -n 4,5p "$U/facts.txt" | tr '\n' ' ')"
  echo "- Rollback to v2.2.0 right after: boot $RB; $(sed -n 2p "$B/facts.txt")"
  echo "- Logits, fresh against upgrade boot: $(sed -n 2p "$RES/logits.txt")"
  echo; echo "Verdict: $V.${why:+ Reasons:$why}"; echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"; cp "$K/comment.md" "$RES/comment.md"
exit 0
