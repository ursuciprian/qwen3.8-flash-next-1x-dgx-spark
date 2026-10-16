#!/usr/bin/env bash
# k106s-1x-ship-check (2026-10-16, 1x #27 #64): release build and fresh-user check boot of 1x v2.4.0 = 1x v2.3.0 with the
# HC down projection BF16 path (b12x a1ad1b69, k100ab WIN) and GDN prefill views (vLLM 10d0acc8, VLLM_GDN_PREFILL_VIEWS=1,
# k101s WIN); together k103 RULE=WIN, k105 c1 recheck PASS, k103g GATE=PASS. Modelled on k104s (2x v2.2.0).
#   image   BASE = the tested image $TESTED (k103/k105/k103g, same Id on both Sparks: CI build-b12x-overlay on the 1x v2.3.0
#           image) + one layer: the torch AOT cache the k103 arm boots compiled for this image (keys 246a576d, 5af689ee,
#           f5d573a4: rank_0_0 + inductor_cache from dgx-01's k103-cb cache; the tested image has no AOT seed at all, and
#           v2.3.0's artifacts under the same keys are for other b12x and vLLM files). Plan 33d9b32e (303 records) is in
#           the tested image, unchanged. Then STATE ci-open: the Mac poster runs CI build-b0-warm on the dgx-01 runner
#           (base BASE, tag $TAG, recipe ref main) and writes ci-done (cap 2 h), as for k104s. 1x-v2.4.0 is added after
#           a pass (retag-image).
#   bytes   fail-closed: TAG's layers start with every layer of TESTED; the vLLM and b12x python trees match TESTED; the
#           six 10d0acc8 files match k101/overlay/patched.sha256 in TESTED and TAG; the image markers say b12x a1ad1b69 and
#           vLLM overlay 10d0acc8; the seed's three AOT models match the k103-cb cache; plan 33d9b32e is TESTED's (sha256,
#           303 records). dgx-02 pulls TAG from ghcr (anonymous) and must get the same Id as dgx-01.
#   check   plain `sparkrun run $REC --hosts <dgx-01> --solo` of the release branch recipe ($REC, sha256 of the code lines
#           checked; the registry follows main, so a by-name registry boot can only happen after the merge) with the real
#           runtime cache: plan 33d9b32e and the new compile cache root $ROOT are moved out first, so the image has to
#           supply them. One boot on dgx-01 (k104s booted once; a 1x boot needs one Spark).
#           pass = AOT loaded from $ROOT (no recompile, no Dynamo transform, nothing saved), 0 b12x plans measured, 303
#           records, views line, KV pool 993,754 tokens, pong, /v1/models, revision + VLLM_GDN_PREFILL_VIEWS=1 +
#           VLLM_CACHE_ROOT, T=0 fresh-c8 acceptance within 0.03 of the k103 dgx-01 arm, llama-benchy pp2048 + tg512 c1/c8
#           (3 runs) not below the k103 dgx-01 arm means by more than the k103 bands.
#   recheck written before any boot (k94b rule): if a benchy cell misses its band on boot 1, one more boot; then PASS
#           needs the 2-boot mean of every cell inside its band and no boot below the lowest k103 arm boot of that cell.
#           Reported, not judged: the k105 dgx-01 views-on arm (same image, a day later than k103: pp2048 1887.0, tg512
#           c1 59.0), because the k103 pp2048 band (1.00%) is tighter than the k103 -> k105 drift of that cell.
#   end     moved items go back unless the boot recreated them (then they stay in $K/moved); lib.sh restores the
#           shipped 2x (v2.1.0 from the registry). Promotion happens outside this job after a pass.
# ETA ~1 h (build ~2, CI ~10-30, boot ~4, checks ~15) + restore ~6 min. Contract: both locks; exit 75 if busy;
# STATE DONE/FAILED; comment.md + RESULT. Nothing is deleted: leftovers of an earlier run are moved aside.
NEEDS="dgx01 dgx02"
set -u
J=k106s; K=$HOME/GEN-AI/backlog/k106s
RES=${RES:-$HOME/GEN-AI/qwen3.8-flash-next-dgx-spark-tp-2/results/k106s-1x-ship-check-$(TZ=Europe/Bucharest date +%Y%m%d-%H%M)}
source "$HOME/GEN-AI/backlog/lib.sh"; source "$BL/k84/cred.sh"   # BENCHY_SRC, TOK, MODEL
TESTED=ghcr.io/ursuciprian/spark-vllm-b12x:k103-20261015-a1ad1b69-10d0acc8; TESTED_ID=sha256:5fb95ca54800ffeb595d0e5f5cde0cb77d63b1bdeabff7151eed353678967e7e
BASE=spark-vllm-b12x:k106s-base-a1ad1b69-5dad364d-10d0acc8
TAG=ghcr.io/ursuciprian/spark-vllm-b12x:k106s-20261017-a1ad1b69-5dad364d-10d0acc8-warm
REV=03f4a0570496bbe741189f4a2a7d611679e85406; REC=$K/recipe.yaml
B12X_SHA=a1ad1b69d491333c00c603b10d1cf46f8bf8fc22; VLLM_SHA=10d0acc85fb450472230558eda063c9059c20234
RECCODE=71a5ee8537865a2e767814d0c8c47d86d711fd05f1aadf691f0101ed5ea06ff1   # sha256 of the recipe without comment lines (release branch at staging)
ROOT=vllm-b12x-a1ad1b69-10d0acc8
PLAN=33d9b32e995b59bc4b0d3cdfb5fc2247f3c7b2151c09569ffd3dbaf85c4562e3.json; NREC=303
# key:sha256 of rank_0_0/model in dgx-01's k103-cb cache (k103 arm boots)
AOTS="246a576d402b4a2d157929f266e8abe046d2f7cb0c64485afc727afabfbd0dfc:4b3ea32b413a3764cd47a090b069faf85821cb6b8bb3066ac9649e66212df541
5af689ee723735c62315581b3870a177e9646d6f86782cb8ec3d0a85660d0b57:8ebe11761d77a054f529e16dd6e3023fc338a4e99f02a12f84ff4d888df46619
f5d573a406e84f4d3353908f74e4b130a7e529c4c8a5abb02791117c1a4034e8:20e0a466bb102e9231123a6b7df83c2118370bb2e3967c6f9a1a17d425530194"
RC=$HOME/.cache/sparkrun/runtime-cache/vllm/ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923
AOTSRC=$RC/k103-cb/vllm/torch_compile_cache/torch_aot_compile
SEED=/opt/b12x-seed/torch_compile_cache/torch_aot_compile
DP=/usr/local/lib/python3.12/dist-packages
PATCHED=$BL/k101/overlay/patched.sha256
VIEWS='GDN prefill views: in-order'
KVL='GPU KV cache size: 993,754 tokens'
CORPUS=$R/results/corpus-code.txt
# k103 dgx-01 arm (results/k103-1x-prefill-combo-ab-20261016-1206/cb-01, 4 boots): mean, band %, lowest arm boot
REF='pp2048 (c1):1915.6:1.00:1898.9;tg512 (c1):58.4:9.04:55.9;tg512 (c8):136.2:7.13:133.0'
REF2='k105 dgx-01 views-on arm (6 boots, reported only): pp2048 (c1) 1887.0, tg512 (c1) 59.0'
ACC_REF="0.849 0.700 0.577 0.464"   # k103 dgx-01 arm, fresh-c8 T=0, pooled over the 4 arm boots
on2() { ssh -n -o ConnectTimeout=10 $H2 "$@"; }
recok() { [ "$(grep -vE '^[[:space:]]*#' "$REC" | sha256sum | cut -c1-64)" = "$RECCODE" ] && grep -qx "container: $TAG" "$REC" \
  && grep -qx "model_revision: $REV" "$REC" && grep -q -- "--revision $REV" "$REC" && grep -qx '  VLLM_GDN_PREFILL_VIEWS: "1"' "$REC" \
  && grep -qx "  VLLM_CACHE_ROOT: \"/cache/runtime/$ROOT\"" "$REC" && grep -q 'cp -rn "$A/." "$VLLM_CACHE_ROOT/torch_compile_cache/"' "$REC" \
  && grep -qx 'solo_only: true' "$REC" && sed -n 1p "$REC" | grep -qx '# Release: v2.4.0'; }
pytree() { docker run --rm --network none --entrypoint bash "$1" -c "cd $DP && find vllm b12x -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64"; }
filesha() { docker run --rm --network none --entrypoint sha256sum "$1" "$2" | cut -c1-64; }
markers() { docker run --rm --network none --entrypoint sh "$1" -c 'cat /workspace/b12x-source-commit /workspace/vllm-overlay-commit' | tr '\n' ' '; }
planrec() { docker run --rm --network none --entrypoint python3 "$1" -c "import json; print(len(json.load(open('/opt/b12x-seed/preparation/$PLAN'))['records']))"; }
srcok() { local e k s; for e in $AOTS; do k=${e%%:*}; s=${e#*:}
  [ "$(sha256sum "$AOTSRC/$k/rank_0_0/model" 2>/dev/null | cut -c1-64)" = "$s" ] && [ -d "$AOTSRC/$k/inductor_cache" ] || return 1; done; }
aside() { [ -e "$1" ] && mv "$1" "$1.old-$(date +%Y%m%d-%H%M%S)"; true; }   # nothing is deleted
if [ "${1:-}" = --dry-run ]; then rc=0
  recok && echo "ok recipe v2.4.0 (code sha256 $RECCODE)" || { echo "recipe check failed"; rc=1; }
  [ "$(docker image inspect -f '{{.Id}}' $TESTED 2>/dev/null)" = $TESTED_ID ] && echo "ok $TESTED on dgx-01" || { echo "$TESTED missing or wrong Id on dgx-01"; rc=1; }
  [ "$(on2 "docker image inspect -f '{{.Id}}' $TESTED" 2>/dev/null)" = $TESTED_ID ] && echo "ok $TESTED on dgx-02" || { echo "$TESTED missing or wrong Id on dgx-02"; rc=1; }
  srcok && echo "ok k103-cb AOT artifacts (3 keys, rank_0_0 + inductor_cache, dgx-01)" || { echo "k103-cb AOT artifacts missing or changed"; rc=1; }
  [ -s "$PATCHED" ] || { echo "missing $PATCHED"; rc=1; }
  for f in check_seed.py depth_decode_probe.py; do [ -s "$R/scripts/$f" ] || { echo "missing $f"; rc=1; }; done
  [ -d "$BENCHY_SRC" ] && [ -d "$TOK" ] && [ -s "$CORPUS" ] || { echo "missing llama-benchy fork, tokenizer or corpus"; rc=1; }
  command -v sparkrun > /dev/null || { echo "missing sparkrun"; rc=1; }
  for f in $PLAN $ROOT; do [ -e "$K/moved/$f" ] && { echo "$K/moved/$f exists (earlier run): move it back or aside first"; rc=1; }; done
  docker image inspect $TAG > /dev/null 2>&1 && echo "note: $TAG already on dgx-01 (a rerun builds and pushes it again)"
  echo "dry-run exit=$rc"; exit $rc; fi
need_lock
others_busy && { echo "another GPU job is active" >&2; exit 75; }
mkdir -p "$RES"; echo "$RES" > "$K/RESULT"; aside "$K/comment.md"; job_begin
MOVED=
unmove() { [ -n "$MOVED" ] || return 0
  ( cd "$RC" && { [ -e b12x/compile/preparation/$PLAN ] || { [ -e "$K/moved/$PLAN" ] && mv "$K/moved/$PLAN" b12x/compile/preparation/; }
    [ -e $ROOT ] || { [ -e "$K/moved/$ROOT" ] && mv "$K/moved/$ROOT" .; }; true; } ); MOVED=; }
trap 'kill $G1 $G2 2>/dev/null; unmove; restore || { sleep 60; restore; } || FINAL="FAILED: restore -- $FINAL"; st "$FINAL"' EXIT
fail() { FINAL="FAILED: k106s $*"; exit 1; }
recok || fail "recipe check"
cp "$K/run.sh" "$REC" "$RES/"
st "running: stop 2x"; stop_all

st "running: build the local base $BASE"
B=$RES/build; mkdir -p "$B"; I=$K/img; aside "$I"; mkdir -p "$I/aot"
srcok || fail "k103-cb AOT artifacts changed since staging"
for e in $AOTS; do k=${e%%:*}; mkdir -p "$I/aot/$k"; cp -a "$AOTSRC/$k/rank_0_0" "$AOTSRC/$k/inductor_cache" "$I/aot/$k/" || fail "copy of AOT key ${k:0:8}"; done
chmod -R u+rwX,go+rX,go-w "$I/aot"
( cd "$I/aot" && find . -type f | LC_ALL=C sort | xargs sha256sum ) > "$B/aot.sha256"
for e in $AOTS; do grep -q "^${e#*:}  ./${e%%:*}/rank_0_0/model$" "$B/aot.sha256" || fail "staged AOT sha256 ($B/aot.sha256)"; done
cat > "$I/Dockerfile" <<EOF
# k106s (1x #27, 2026-10-16): the k103/k105/k103g test image ($TESTED, Id ${TESTED_ID:7:12}) + the torch AOT cache
# (keys 246a576d, 5af689ee, f5d573a4) of the k103 arm boots, compiled for its b12x and vLLM files. Plan 33d9b32e is TESTED's.
FROM $TESTED
COPY aot/ $SEED/
EOF
cp "$I/Dockerfile" "$B/"
DOCKER_BUILDKIT=1 timeout 1200 docker build -t "$BASE" "$I" > "$B/build.log" 2>&1 9>&- || fail "docker build ($B/build.log)"

layers() { docker image inspect -f '{{json .RootFS.Layers}}' "$1"; }
prefix() { python3 -c 'import json,sys; a,b=json.loads(sys.argv[1]),json.loads(sys.argv[2]); sys.exit(0 if b[:len(a)]==a and len(b)>len(a) else 1)' "$(layers "$1")" "$(layers "$2")"; }
PLANT=$(filesha "$TESTED" /opt/b12x-seed/preparation/$PLAN)
seedok() { local e; for e in $AOTS; do [ "$(filesha "$1" $SEED/${e%%:*}/rank_0_0/model)" = "${e#*:}" ] || return 1; done
  [ -n "$PLANT" ] && [ "$(filesha "$1" /opt/b12x-seed/preparation/$PLAN)" = "$PLANT" ] && [ "$(planrec "$1")" = $NREC ]; }
prefix "$TESTED" "$BASE" && seedok "$BASE" || fail "local base check (layers or seed)"

st "ci-open"   # the Mac poster reads this exact line, triggers build-b0-warm and writes ci-done
aside "$K/ci-done"; echo "$BASE $TAG" > "$K/ci-request"; log "ci window: $BASE -> $TAG"
s=$(date +%s); until [ -f "$K/ci-done" ] || [ $(( $(date +%s) - s )) -gt 7200 ]; do sleep 20; done
log "ci window closed: $(cat "$K/ci-done" 2>/dev/null || echo timeout)"
cp "$K/ci-done" "$B/ci-done.txt" 2>/dev/null
grep -q ' ok ' "$K/ci-done" 2>/dev/null || fail "CI build-b0-warm: $(cat "$K/ci-done" 2>/dev/null || echo 'no ci-done in 2 h')"
DIG=$(docker image inspect -f '{{join .RepoDigests " "}}' "$TAG" 2>/dev/null | tr ' ' '\n' | grep '^ghcr.io/ursuciprian/spark-vllm-b12x@' | head -1)
[ -n "$DIG" ] || fail "no ghcr RepoDigest on $TAG after the CI run"
ID1=$(docker image inspect -f '{{.Id}}' "$TAG")

st "running: byte checks of $TAG against $TESTED"
VB=$B/bytes.txt; : > "$VB"; bad=
prefix "$TESTED" "$TAG" && echo "layers: all $(layers "$TESTED" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))') layers of $TESTED are the first layers of $TAG" >> "$VB" || bad="$bad layers;"
t1=$(pytree "$TESTED"); t2=$(pytree "$TAG"); echo "vllm+b12x python tree sha256: tested $t1, release $t2" >> "$VB"; [ -n "$t1" ] && [ "$t1" = "$t2" ] || bad="$bad pytree;"
while read -r h f; do r=$(filesha "$TAG" "$DP/$f"); rt=$(filesha "$TESTED" "$DP/$f"); echo "$f: release $r, tested $rt (patched.sha256 $h)" >> "$VB"
  [ "$r" = "$h" ] && [ "$rt" = "$h" ] || bad="$bad $f;"; done < "$PATCHED"
m=$(markers "$TAG"); echo "markers (b12x-source-commit vllm-overlay-commit): $m" >> "$VB"; [ "$m" = "$B12X_SHA $VLLM_SHA " ] || bad="$bad markers;"
seedok "$TAG" && echo "seed: AOT $(for e in $AOTS; do printf '%s %s, ' ${e:0:8} $(echo ${e#*:} | cut -c1-12); done)plan ${PLAN:0:8} ${PLANT:0:16} ($NREC records, same as tested)" >> "$VB" || bad="$bad seed;"
[ -z "$bad" ] || fail "byte checks:$bad ($VB)"
st "running: pull $TAG on dgx-02 from ghcr"
on2 "docker pull -q $TAG" > "$B/pull-dgx02.txt" 2>&1 || fail "pull on dgx-02 ($B/pull-dgx02.txt)"
ID2=$(on2 "docker image inspect -f '{{.Id}}' $TAG"); [ "$ID1" = "$ID2" ] || fail "Id differs: dgx-01 $ID1, dgx-02 $ID2"
echo "digest $DIG; Id $ID1 on both Sparks" >> "$VB"

st "running: move the plan file and the $ROOT compile cache root out of the runtime cache"
mkdir -p "$K/moved"
for f in $PLAN $ROOT; do [ -e "$K/moved/$f" ] && fail "$K/moved/$f exists from an earlier run"; done
( cd "$RC" && { [ -e b12x/compile/preparation/$PLAN ] && mv b12x/compile/preparation/$PLAN "$K/moved/" && echo moved plan
  [ -e $ROOT ] && mv $ROOT "$K/moved/" && echo moved $ROOT; true; } ) > "$RES/cleared.txt" 2>&1; MOVED=1

sc() { docker ps -a --format '{{.Names}}' | grep -E '_solo|sparkrun' | head -1; }
pos() { curl -s -m 10 "localhost:8000/metrics" | grep -E '^vllm:spec_decode_num_(accepted|draft)_tokens(_per_pos)?_total'; }
boot() { # n: boot the release recipe on dgx-01, collect facts into $D (= $RES/boot$n)
  D=$RES/boot$1; mkdir -p "$D"; local s=$(date +%s) c; up=
  ( cd "$K" && timeout -k 60 1800 sparkrun run "$REC" --hosts $H1 --solo --no-follow ) > "$D/sparkrun.log" 2>&1 < /dev/null 9>&-
  while [ $(( $(date +%s) - s )) -lt 1800 ]; do [ "$(health localhost)" = 200 ] && { up=$(( $(date +%s) - s )); break; }
    [ $(( $(date +%s) - s )) -gt 240 ] && [ -z "$(docker ps -q --filter name=sparkrun)" ] && break; sleep 5; done
  c=$(sc); docker exec "$c" cat /tmp/sparkrun_serve.log > "$D/serve.log" 2>&1 || docker logs "$c" > "$D/serve.log" 2>&1
  [ -n "$up" ] || return 1
  docker exec "$c" sh -c "ps aux | grep '[v]llm serve'" > "$D/cmdline.txt" 2>&1
  docker exec "$c" env | grep -E '^(VLLM_GDN_PREFILL_VIEWS|VLLM_CACHE_ROOT|XDG_CACHE_HOME)=' | sort > "$D/env.txt" 2>&1
  docker inspect --format '{{.Config.Image}} {{.Image}}' "$c" > "$D/image.txt" 2>&1
  P=$(pong localhost); curl -s -m 20 localhost:8000/v1/models > "$D/models.json"
  echo "boot $1: health after $up s" > "$D/boot.txt"; return 0; }
warm() { grep -q "Directly load AOT compilation from path /cache/runtime/$ROOT/" "$D/serve.log" \
  && ! grep -qE 'Compiling model again|saved AOT compiled function|Dynamo bytecode transform' "$D/serve.log" && echo WARM || echo COLD; }
benchy() { ( cd "$R" && child timeout -k 30 1800 uvx --from "$BENCHY_SRC" llama-benchy --base-url "http://$H1:8000/v1" --model $MODEL \
    --tokenizer "$TOK" --prompt-mode task --no-force-length --pp 2048 --tg 512 --depth 0 --concurrency 1 8 --runs 3 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --enable-prefix-caching --metrics-url "http://$H1:8000/metrics" \
    --save-result "$D/task.csv" > "$D/task.log" 2>&1 ); }
judge() { # task.csv files -> OK|BAD | text; mean of the boots given vs the k103 dgx-01 arm
  python3 - "$REF" "$@" <<'PY'
import sys
ref = {}
for p in sys.argv[1].split(";"):
    name, m, band, lo = p.rsplit(":", 3); ref[name] = (float(m), float(band), float(lo))
runs = []
for path in sys.argv[2:]:
    got = {}
    for line in open(path):
        f = [x.strip() for x in line.split("|")]
        if len(f) > 3 and f[2] in ref:
            try: got[f[2]] = float(f[3].split()[0])
            except ValueError: pass
    runs.append(got)
out, bad = [], []
for name, (m, band, lo) in ref.items():
    vals = [r[name] for r in runs if name in r]
    if len(vals) != len(runs): bad.append(f"{name} missing"); continue
    v = sum(vals) / len(vals); d = (v / m - 1) * 100
    out.append(f"{name} {'/'.join(f'{x:.1f}' for x in vals)} -> {v:.1f} ({d:+.1f}% vs k103 arm {m}, band {band}%)")
    if d < -band: bad.append(f"{name} below band")
    if len(runs) > 1 and min(vals) < lo - 0.05: bad.append(f"{name} boot below the lowest arm boot {lo}")
print(("OK" if not bad else "BAD (" + "; ".join(bad) + ")") + " | " + "; ".join(out))
PY
}

st "running: check boot 1"
boot 1 || fail "check boot: no /health in 1800 s ($D)"
B1=$D; P1=$P; up1=$up; WARM=$(warm)
N1=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["records"]))' $RC/b12x/compile/preparation/$PLAN 2>/dev/null)
SEEDCHK=$(python3 "$R/scripts/check_seed.py" "$D/serve.log" 2>&1 | tail -2 | tr '\n' ' ')
MODELS=$(python3 -c 'import json,sys; print(" ".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' "$D/models.json" 2>/dev/null)
st "running: fresh-c8 acceptance probe + llama-benchy (boot 1)"
( cd "$R" && timeout -k 30 300 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label warm --corpus "$CORPUS" --concurrency 1 --repeats 1 --temperature 0 --out "$D/probe-warm.json" --depth 4096 --new 2048 --offset 900000 --max-tokens 512 > "$D/probe-warm.log" 2>&1 )
pos > "$D/pos.before"
( cd "$R" && timeout -k 30 500 python3 scripts/depth_decode_probe.py --base "http://$H1:8000" --label fresh-c8 --corpus "$CORPUS" --concurrency 8 --repeats 2 --temperature 0 --out "$D/probe-fresh-c8.json" --depth 256 --new 2048 --offset 418916 --max-tokens 512 > "$D/probe-fresh-c8.log" 2>&1 )
pos > "$D/pos.after"
ACC=$(python3 - "$D/pos.before" "$D/pos.after" <<'PY'
import re, sys
def rd(p):
    c = {"accepted": {}, "draft": {}}
    for l in open(p):
        m = re.match(r'vllm:spec_decode_num_(accepted|draft)_tokens_per_pos_total\{.*position="(\d+)"\} ([0-9.e+]+)', l)
        if m: c[m[1]][int(m[2])] = c[m[1]].get(int(m[2]), 0) + float(m[3])
    return c
a, b = rd(sys.argv[1]), rd(sys.argv[2])
print(" ".join(f"{(b['accepted'].get(k,0)-a['accepted'].get(k,0))/max(b['draft'][k]-a['draft'].get(k,0),1):.3f}" for k in sorted(b["draft"])))
PY
)
benchy; BV=$(judge "$B1/task.csv")
{ echo "image: $(tr '\n' ' ' < "$D/image.txt")"; echo "boot 1: health after $up1 s; pong: $P1; /v1/models: $MODELS"
  f=serve.log; echo "$f: Directly load AOT $(grep -c 'Directly load AOT compilation' "$D/$f") (from $ROOT $(grep -c "Directly load AOT compilation from path /cache/runtime/$ROOT/" "$D/$f")), compiled again $(grep -c 'Compiling model again' "$D/$f"), saved AOT $(grep -c 'saved AOT compiled function' "$D/$f"), Dynamo bytecode transform $(grep -c 'Dynamo bytecode transform' "$D/$f"), views lines $(grep -c "$VIEWS" "$D/$f"), KV 993,754 lines $(grep -c "$KVL" "$D/$f"), MXFP8 copies $(grep -c 'MXFP8 copy serves rows >= 41 ' "$D/$f"), Traceback $(grep -c Traceback "$D/$f")"
  echo "$SEEDCHK"; echo "plan $PLAN: ${N1:-missing} records (seed $NREC)"
  echo "revision: $(grep -oE -- "--revision [0-9a-f]+" "$D/cmdline.txt" | head -1); env: $(tr '\n' ' ' < "$D/env.txt")"
  sed 's/^/cleared before boot: /' "$RES/cleared.txt"; } > "$D/facts.txt"
V=PASS; why=
[ "$WARM" = WARM ] || { V=FAIL; why="$why boot $WARM;"; }
echo "$SEEDCHK" | grep -q 'seed check PASS' && [ "$N1" = $NREC ] || { V=FAIL; why="$why plans measured or records;"; }
grep -q "$VIEWS" "$D/serve.log" || { V=FAIL; why="$why views line missing;"; }
grep -q "$KVL" "$D/serve.log" || { V=FAIL; why="$why KV pool;"; }
echo "$P1" | grep -qi pong || { V=FAIL; why="$why no pong;"; }
echo " $MODELS " | grep -q ' qwen3.8-flash-next ' || { V=FAIL; why="$why /v1/models;"; }
grep -q -- "--revision $REV" "$D/cmdline.txt" && grep -qx 'VLLM_GDN_PREFILL_VIEWS=1' "$D/env.txt" && grep -qx "VLLM_CACHE_ROOT=/cache/runtime/$ROOT" "$D/env.txt" || { V=FAIL; why="$why cmdline or env;"; }
python3 -c 'import sys; a=[float(x) for x in sys.argv[1].split()]; r=[float(x) for x in sys.argv[2].split()]; sys.exit(0 if len(a)>=4 and all(abs(x-y)<=0.03 for x,y in zip(a,r)) else 1)' "$ACC" "$ACC_REF" || { V=FAIL; why="$why acceptance $ACC;"; }
BOOTS="1"; BJ=$BV
if [ "$V" = PASS ] && [ "${BV%% *}" != OK ]; then   # pre-registered recheck (header): one more boot, judge the 2-boot mean
  log "boot 1 benchy outside a band ($BV): recheck boot"; stop_all
  st "running: recheck boot 2"
  if boot 2; then w2=$(warm); B2=$D; benchy; BJ=$(judge "$B1/task.csv" "$B2/task.csv"); BOOTS="1 2"
    echo "$P" | grep -qi pong || { V=FAIL; why="$why boot 2 no pong;"; }
    grep -q "$VIEWS" "$D/serve.log" || { V=FAIL; why="$why boot 2 views line;"; }
    echo "boot 2: health after $up s, AOT $w2" >> "$B1/facts.txt"
  else V=FAIL; why="$why recheck boot did not come up;"; fi
fi
[ "${BJ%% *}" = OK ] || { V=FAIL; why="$why ${BJ%% |*};"; }
docker exec "$(sc)" cat /tmp/sparkrun_serve.log > "$D/serve-end.log" 2>&1; stop_all
{ echo "k106s $(TZ=Europe/Bucharest date '+%F %T %Z'): 1x v2.4.0 build and fresh-user check boot, image $TAG: $V"
  echo "digest: ${DIG#*@}"; cat "$B/bytes.txt"; cat "$B1/facts.txt"
  echo "fresh-c8 acceptance per position (T=0): $ACC (k103 dgx-01 arm $ACC_REF)"; echo "benchy (boots $BOOTS): $BJ"
  echo "reference, not judged: $REF2"; } > "$RES/k106s.txt"
FINAL="DONE: k106s $V boot=$WARM ${up1}s digest=${DIG#*@} acc=$ACC ($RES/k106s.txt)${why:+ --$why}"
{ echo "k106s-1x-ship-check result: I built the 1x v2.4.0 release image and booted it the way a new user would, with a plain \`sparkrun run\` of the release recipe on one Spark (1x v2.3.0 plus the HC down projection BF16 path, \`VLLM_GDN_PREFILL_VIEWS=1\` and a compile cache root of its own)."
  echo; echo "- Image \`$TAG\`, digest \`${DIG#*@}\`. It is the image the A/B, the c1 recheck and the gate ran on, plus one layer with the torch AOT cache of the A/B arm boots. Every layer of the tested image is in it unchanged, and the vLLM and b12x code is byte-identical to the tested image. The second Spark pulled it from ghcr and got the same image."
  echo "- Before the boot I moved the plan file and the new compile cache root out of the runtime cache, so the image had to supply them. Up in $up1 s, $WARM (compile cache loaded from the image, no compile), $(echo "$SEEDCHK" | grep -oE 'seed check (PASS|FAIL)'), plan records $N1. The GDN views path is on and the KV pool is 993,754 tokens."
  echo "- Pong and \`/v1/models\` answer; the server runs \`--revision ${REV:0:8}\`."
  echo "- T=0 fresh-c8 acceptance per draft position: $ACC (A/B arm: $ACC_REF)."
  echo "- llama-benchy (T=1, 3 runs, boots $BOOTS) against the A/B arm means: ${BJ#*| }. For reference, not judged: the c1 recheck arm on the same image had pp2048 c1 1887.0 and tg512 c1 59.0."
  echo; echo "Verdict: $V.${why:+ Reasons:$why} The quality gate is k103g (same image code, weights, plans and flag)."
  echo; echo "Results: {RESULTS_URL}"; } > "$K/comment.md"
sed -i -e 's/\xe2\x80\x94/-/g' -e 's/\xe2\x80\x93/-/g' -e 's/\xc2\xb7/-/g' "$K/comment.md"; cp "$K/comment.md" "$RES/comment.md"
exit 0
