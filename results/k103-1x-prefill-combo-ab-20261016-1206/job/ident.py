#!/usr/bin/env python3
"""k101 quality identity (stdlib): greedy completions of fixed prompts, compared across boots.

  ident.py run --base URL --corpus FILE --out OUT.json     8 prompts (corpus windows of 1K..24K chars plus a question),
                                                           temperature 0, 96 new tokens, one at a time
  ident.py report <node dir>                               reads <node>/{ctl,arm}-p*/hook/ident.json
  ident.py selftest

The views path runs the same kernels as the gather path, so a prompt whose completion every control boot reproduces
must come out the same on every arm boot. Prompts where the control boots disagree among themselves (nondeterministic
reductions) are counted but not judged. Last line IDENT=PASS|FAIL|UNSTABLE (no control-stable prompt)|EMPTY.
"""
import glob
import json
import os
import sys
import urllib.request

MODEL = "qwen3.8-flash-next"
WINDOWS = ((0, 1000), (5_000, 3000), (40_000, 6000), (90_000, 12000), (150_000, 24000),
           (260_000, 2000), (300_000, 9000), (420_000, 16000))
ASK = "\n\nIn two sentences, what does the code above do?"


def run(base, corpus, out):
    text = open(corpus, errors="replace").read()
    res = []
    for i, (off, n) in enumerate(WINDOWS):
        body = {"model": MODEL, "prompt": f"[{i}] " + text[off:off + n] + ASK, "max_tokens": 96,
                "temperature": 0, "seed": 0}
        req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                res.append(json.load(r)["choices"][0]["text"])
        except Exception as e:  # a failed request is a missing cell, never a match
            res.append(None)
            print(f"prompt {i}: {e}", file=sys.stderr)
    json.dump({"texts": res}, open(out, "w"))
    print(f"ident: {sum(t is not None for t in res)}/{len(res)} completions")


def report(node):
    boots = {}
    for who in ("ctl", "arm"):
        for p in sorted(glob.glob(os.path.join(node, f"{who}-p*", "hook", "ident.json"))):
            boots.setdefault(who, []).append(json.load(open(p))["texts"])
    ctl, arm = boots.get("ctl", []), boots.get("arm", [])
    if len(ctl) < 2 or not arm:
        return [f"ident: {len(ctl)} control / {len(arm)} arm boots"], "EMPTY"
    stable = same = unstable = 0
    lines = []
    for i in range(len(WINDOWS)):
        c = [b[i] for b in ctl]
        a = [b[i] for b in arm]
        if None in c or None in a:
            lines.append(f"  prompt {i}: missing completion"); return lines, "EMPTY"
        if len(set(c)) > 1:
            unstable += 1
            lines.append(f"  prompt {i}: control boots disagree ({len(set(c))} variants), arm matches one: "
                         f"{all(x in c for x in a)}")
            continue
        stable += 1
        ok = all(x == c[0] for x in a)
        same += ok
        lines.append(f"  prompt {i}: control stable, arm {'identical' if ok else 'DIFFERENT'}")
    lines.insert(0, f"ident: {stable} prompts stable across {len(ctl)} control boots, arm identical on {same}; "
                    f"{unstable} control-unstable (not judged)")
    # no control-stable prompt: T=0 completions do not reproduce boot to boot on this stack (k101 stage 1 saw
    # 8/8 differ between two control boots), so identity is reported, not judged
    return lines, ("UNSTABLE" if not stable else "PASS" if same == stable else "FAIL")


def selftest():
    import tempfile
    t = tempfile.mkdtemp()
    def put(b, texts):
        os.makedirs(os.path.join(t, b, "hook"), exist_ok=True)
        json.dump({"texts": texts}, open(os.path.join(t, b, "hook", "ident.json"), "w"))
    base = [f"t{i}" for i in range(len(WINDOWS))]
    put("ctl-p1", base); put("ctl-p2", base[:7] + ["x"]); put("arm-p1", base); put("arm-p2", base)
    lines, v = report(t)
    assert v == "PASS" and "7 prompts stable" in lines[0], lines
    put("arm-p2", ["y"] + base[1:])
    assert report(t)[1] == "FAIL"
    put("arm-p2", base[:3] + [None] + base[4:])
    assert report(t)[1] == "EMPTY"
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["report"] and len(a) == 2:
        lines, v = report(a[1]); print("\n".join(lines)); print(f"IDENT={v}")
    elif a[:1] == ["run"]:
        kw = dict(zip(a[1::2], a[2::2]))
        run(kw["--base"], kw["--corpus"], kw["--out"])
    else:
        sys.exit(__doc__)
