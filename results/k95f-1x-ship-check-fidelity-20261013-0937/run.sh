#!/usr/bin/env bash
# k95f-1x-ship-check-fidelity (2026-10-13, 1x #27): the long-context half of the 1x v2.3.0 gate on dgx-02, plus a
# second fresh-user boot (job-local sparkrun HOME, empty runtime cache, image copied by sparkrun from dgx-01).
#   gate B  fidelity_probe 8k/32k/64k/128k + 128k seeds 11 and 13: 20/20 exact everywhere, no near/wrong/explore/no_call.
# ETA ~50 min. Contract (backlog/runner.sh): dgx02 lock; exit 75 if busy; STATE DONE/FAILED; comment.md + RESULT.
NEEDS="dgx02"
set -u
J=k95f; K=$HOME/GEN-AI/backlog/k95f
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k95f-1x-ship-check-fidelity-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"; source "$BL/k95/common.sh"
if [ "${1:-}" = --dry-run ]; then rc=0
  recok && echo "ok recipe v2.3.0" || { echo "recipe check failed"; rc=1; }
  docker image inspect $TAG >/dev/null 2>&1 && echo "ok image $TAG on dgx-01" || { echo "image missing"; rc=1; }
  [ -s "$R/scripts/fidelity_probe.py" ] || { echo "fidelity_probe.py missing"; rc=1; }
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active" >&2; exit 75; }
mkdir -p "$RES"; echo "$RES" > "$K/RESULT"; rm -f "$K/comment.md"; job_begin
recok || { FINAL="FAILED: k95f recipe check"; exit 1; }
cp "$K/run.sh" "$REC" "$RES/"
st "running: stop dgx-02"; stop_all
st "running: fresh-user boot on dgx-02"
FH=$K/fresh-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M); mkhome "$FH"
F=$RES/fresh; sboot "$F" $H2 "$FH" || { FINAL="FAILED: k95f fresh boot ($F)"; exit 1; }
FW=$(facts "$F" "$FH/runtime-cache/vllm/$RCN/b12x/compile/preparation/$PLAN" $H2); log "fresh dgx-02: $FW"
st "running: gate B (fidelity 8k-128k + 128k seeds 11/13)"
G=$F/gate; mkdir -p "$G"
( cd "$R" && child timeout 5400 python3 scripts/fidelity_probe.py --base http://$H2:8000 --model $MODEL --depths 8000,32000,64000,128000 --out "$G/fidelity.json" > "$G/fidelity_probe.txt" 2>&1 )
for sd in 11 13; do ( cd "$R" && child timeout 2700 python3 scripts/fidelity_probe.py --base http://$H2:8000 --model $MODEL --depths 128000 --seed $sd --out "$G/fidelity-seed$sd.json" > "$G/fidelity-seed$sd.txt" 2>&1 ); done
serve_log $H2 "$F/serve-end.log"; stop_all
GB=$(python3 - "$G" <<'PY'
import re, sys
g = sys.argv[1]; rd = lambda f: open(f"{g}/{f}", errors="replace").read() if __import__("os").path.exists(f"{g}/{f}") else ""
fid = re.findall(r"^depth\s+(\d+) .*?exact\s+(\d+)\s+near\s+(\d+)\s+wrong\s+(\d+)\s+explore\s+(\d+)\s+no_call\s+(\d+)", rd("fidelity_probe.txt") + rd("fidelity-seed11.txt") + rd("fidelity-seed13.txt"), re.M)
ok = len(fid) == 6 and all(int(x[1]) == 20 and not any(int(y) for y in x[2:]) for x in fid)
print(("PASS" if ok else "FAIL") + ": " + " ".join(f"{x[0]}:{x[1]}/20" for x in fid))
PY
)
{ echo "k95f $(TZ=Europe/Bucharest date '+%F %T %Z'): 1x v2.3.0 gate B on dgx-02, image $TAG"
  echo "== fresh-user boot (empty runtime cache): $FW"; cat "$F/facts.txt"
  echo "== gate B: $GB"; grep -h "^depth" "$G/fidelity_probe.txt" "$G"/fidelity-seed1?.txt; } > "$RES/k95f.txt"
V=PASS; [ "${GB%%:*}" = PASS ] || V=FAIL
FINAL="DONE: k95f $V fresh=$FW gateB=$GB ($RES/k95f.txt)"
{ echo "k95f result: the long-context half of the 1x v2.3.0 gate, on the other Spark, after a second fresh-user boot (new sparkrun HOME, empty runtime cache): $(sed -n 1p "$F/boot.txt" | sed 's/boot_s /up in /') s, $FW."
  echo; echo "Long-context retrieval (8k, 32k, 64k, 128k, plus 128k seeds 11 and 13, 20 questions each): $GB."
  echo; echo '```'; grep -h "^depth" "$G/fidelity_probe.txt" "$G"/fidelity-seed1?.txt; echo '```'
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
cp "$K/comment.md" "$RES/comment.md"
exit 0
