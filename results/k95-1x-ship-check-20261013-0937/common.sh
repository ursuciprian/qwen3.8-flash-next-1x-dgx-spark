# k95/k95f shared helpers (2026-10-13): 1x v2.3.0 ship check. Sourced after lib.sh and k84/cred.sh.
TAG=ghcr.io/ursuciprian/spark-vllm-b12x:k95-20261013-b45fe867-5dad364d-warm
REV=03f4a0570496bbe741189f4a2a7d611679e85406
PLAN=33d9b32e995b59bc4b0d3cdfb5fc2247f3c7b2151c09569ffd3dbaf85c4562e3.json; NREC=303
REC=$BL/k95/recipe.yaml; SR=$HOME/.local/bin/sparkrun; RHOME=$HOME
RCN=ursuciprian__Qwen3.8-Flash-Next-NVFP4-GDN-MSE-5d63a923
CORPUS=$R/results/corpus-code.txt
recok() { grep -qx "container: $TAG" "$REC" && grep -qx "model_revision: $REV" "$REC" && grep -q -- "--revision $REV" "$REC" \
  && grep -qx -- "    --enable-prompt-tokens-details \\\\" "$REC" && sed -n 1p "$REC" | grep -qx '# Release: v2.3.0'; }
mkhome() { local f=$1; mkdir -p "$f/.config/sparkrun" "$f/sparkrun-cache" "$f/runtime-cache"
  cp "$RHOME/.config/sparkrun/config.yaml" "$f/.config/sparkrun/config.yaml"
  printf 'cache_dir: %s\nhf_cache_dir: %s\nruntime_cache:\n  dir: %s\n' "$f/sparkrun-cache" "$RHOME/.cache/huggingface" "$f/runtime-cache" >> "$f/.config/sparkrun/config.yaml"; }
sboot() { # dir host fresh_home|"" -> 0 when /health 200 and pong; boot.txt
  local d=$1 h=$2 fh=$3 s=$(date +%s) a; mkdir -p "$d"; cp "$REC" "$d/recipe.yaml"
  if [ -n "$fh" ]; then ( cd "$fh" && timeout -k 60 1800 env HOME=$fh HF_HOME=$RHOME/.cache/huggingface "$SR" run "$REC" --hosts $h --solo --no-follow ) > "$d/sparkrun.log" 2>&1 < /dev/null 9>&-
  else ( cd "$(dirname "$REC")" && timeout -k 60 1800 "$SR" run "$REC" --hosts $h --solo --no-follow ) > "$d/sparkrun.log" 2>&1 < /dev/null 9>&-; fi
  while [ "$(health $h)" != 200 ]; do a=$(( $(date +%s) - s ))
    [ $a -gt 2400 ] && { serve_log $h "$d/serve.log"; log "$d: no /health after $a s"; return 1; }
    if [ $a -gt 240 ] && [ -z "$(x $h "docker ps -q --filter name=sparkrun")" ]; then serve_log $h "$d/serve.log"; log "$d: container gone after $a s"; return 1; fi
    sleep 5; done
  a=$(( $(date +%s) - s )); serve_log $h "$d/serve.log"; echo "boot_s $a" > "$d/boot.txt"; echo "pong $(pong $h)" >> "$d/boot.txt"
  x $h "docker exec \$(docker ps --format '{{.Names}}' | grep sparkrun | head -1) sh -c \"ps aux | grep '[v]llm serve'\"" > "$d/cmdline.txt" 2>&1
  x $h "docker inspect \$(docker ps -q --filter name=sparkrun | head -1) --format '{{.Config.Image}} {{range .Mounts}}{{.Source}}->{{.Destination}} {{end}}'" > "$d/mounts.txt" 2>&1
  log "$d: up after $a s, $(sed -n 2p "$d/boot.txt")"; grep -qi pong "$d/boot.txt"; }
facts() { # dir plan_file_path host -> facts.txt; prints WARM|COLD
  local d=$1 f=$2 h=$3 n
  n=$(x $h "python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))[\"records\"]))' $f" 2>/dev/null)
  { echo "image: $(cut -d' ' -f1 "$d/mounts.txt")"; echo "runtime cache: $(tr ' ' '\n' < "$d/mounts.txt" | grep -- '->/cache/runtime$')"
    echo "boot: $(tr '\n' ' ' < "$d/boot.txt")"
    echo "Directly load AOT: $(grep -c 'Directly load AOT compilation' "$d/serve.log"); saved AOT: $(grep -c 'saved AOT compiled function' "$d/serve.log"); Dynamo bytecode transform: $(grep -c 'Dynamo bytecode transform' "$d/serve.log")"
    python3 "$R/scripts/check_seed.py" "$d/serve.log" 2>&1 | tail -2
    echo "plan $PLAN: ${n:-missing} records (seed $NREC)"
    echo "MXFP8 copies: $(grep -c 'MXFP8 copy serves rows >= 41 ' "$d/serve.log")"
    echo "Traceback: $(grep -c Traceback "$d/serve.log"); ValueError: $(grep -c ValueError "$d/serve.log"); 'plan belongs to': $(grep -c 'plan belongs to' "$d/serve.log")"
    echo "revision: $(grep -oE -- "--revision [0-9a-f]+" "$d/cmdline.txt" | head -1); prompt tokens details flag: $(grep -c -- '--enable-prompt-tokens-details' "$d/cmdline.txt")"; } > "$d/facts.txt"
  if grep -q 'Directly load AOT' "$d/serve.log" && ! grep -qE 'saved AOT compiled function|Dynamo bytecode transform' "$d/serve.log" \
     && grep -q 'seed check PASS' "$d/facts.txt" && [ "$n" = $NREC ]; then echo WARM; else echo COLD; fi; }
cached() { # host -> "prompt_tokens=N cached_tokens=M" of the second of two identical ~12K-token prompts
  python3 - "http://$1:8000" "$CORPUS" <<'PY'
import json, sys, urllib.request
base, corpus = sys.argv[1:3]
text = open(corpus, errors="replace").read()[:48000]
body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": text + "\n\nIn one word: which language is most of the code above?"}],
        "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
u = None
for _ in range(2):
    r = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(r, timeout=600))["usage"]
d = u.get("prompt_tokens_details") or {}
print(f"prompt_tokens={u['prompt_tokens']} cached_tokens={d.get('cached_tokens')}")
PY
}
