#!/usr/bin/env python3
"""k74 cold prefill client (2026-10-08): prompt tokens / time to first token on prompts that share nothing with the
prefix cache. Each request is 16 seeded random token ids (unique per length and repeat) followed by its own window of
corpus tokens (distinct windows, tokenized through the server's /tokenize), sent to /v1/completions as token ids with
max_tokens 1, temperature 0, streamed. TTFT = time to the first streamed chunk. Rate per length = prompt tokens /
mean TTFT over the repeats (same as the per-request numbers, which are kept too).

  cold.py run --base URL --corpus FILE --lens 16384,65536,131072 --reps 3 --seed p1 --out OUT.json
  cold.py report <thunderdome node dir>...   ctl vs arm from <dir>/{ctl,arm}-p{1,2}/hook/cold.json
  cold.py selftest
"""
import json, random, statistics, sys, time, urllib.request

MODEL = "qwen3.8-flash-next"
NONCE = 16
SLICE = 120_000  # chars per /tokenize call (~30K tokens)


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def nonce(seed, n, r):
    rng = random.Random(f"{seed}|{n}|{r}")
    return [rng.randrange(1000, 100000) for _ in range(NONCE)]


def plan(lens, reps, seed, ids):
    """[(len, rep, prompt ids)]: nonce + a distinct window of ids per request (wraps if the corpus is short)."""
    out, cur = [], 0
    for n in lens:
        for r in range(reps):
            need = n - NONCE
            w = [ids[(cur + i) % len(ids)] for i in range(need)]
            cur += need
            out.append((n, r, nonce(seed, n, r) + w))
    return out


def tokenize(base, text, need):
    ids, pos = [], 0
    while len(ids) < need and pos < len(text):
        with post(base, "/tokenize", {"model": MODEL, "prompt": text[pos:pos + SLICE], "add_special_tokens": False}) as r:
            ids += json.load(r)["tokens"]
        pos += SLICE
    return ids


def one(base, prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 1, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time(); ttft = ptok = None
    with post(base, "/v1/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ttft = ttft or time.time() - t0
            u = json.loads(line[6:]).get("usage")
            if u:
                ptok = u.get("prompt_tokens")
    return {"ttft_s": ttft, "prompt_tokens": ptok or len(prompt)}


def summary(rows):
    out = {}
    for n in sorted({r["len"] for r in rows}):
        rr = [r for r in rows if r["len"] == n and r.get("ttft_s")]
        if not rr:
            out[str(n)] = {"n": 0}; continue
        t = [r["ttft_s"] for r in rr]; p = statistics.mean(r["prompt_tokens"] for r in rr)
        out[str(n)] = {"n": len(rr), "prompt_tokens": p, "ttft_mean_s": statistics.mean(t), "ttft_min_s": min(t),
                       "tok_s": p / statistics.mean(t), "tok_s_each": [round(r["prompt_tokens"] / r["ttft_s"]) for r in rr]}
    return out


def run(a):
    lens = [int(x) for x in a["lens"].split(",")]; reps = int(a["reps"])
    text = open(a["corpus"], errors="replace").read()
    ids = tokenize(a["base"], text, sum(n - NONCE for n in lens) * reps)
    rows = []
    for n, r, prompt in plan(lens, reps, a["seed"], ids):
        try:
            row = one(a["base"], prompt)
        except Exception as e:  # keep going: a failed request is reported, not fatal
            row = {"ttft_s": None, "prompt_tokens": None, "error": repr(e)[:300]}
        row.update(len=n, rep=r); rows.append(row)
        print(json.dumps(row), flush=True)
    res = {"seed": a["seed"], "corpus_tokens": len(ids), "rows": rows, "summary": summary(rows)}
    json.dump(res, open(a["out"], "w"), indent=1)
    for n, s in res["summary"].items():
        print(f"{n}: " + (f"n={s['n']} prompt {s['prompt_tokens']:.0f} tok, TTFT mean {s['ttft_mean_s']:.2f} s "
                         f"min {s['ttft_min_s']:.2f} s, {s['tok_s']:.0f} tok/s (each {s['tok_s_each']})" if s["n"] else "no data"))


def report(dirs):
    for d in dirs:
        print(f"== {d.rstrip('/').split('/')[-1]}: cold prefill tok/s per boot (hook, prompt tokens / mean TTFT, 3 reps)")
        per = {}
        for b in ("ctl-p1", "arm-p1", "arm-p2", "ctl-p2"):
            try:
                s = json.load(open(f"{d}/{b}/hook/cold.json"))["summary"]
            except (OSError, ValueError, KeyError):
                continue
            for n, v in s.items():
                if v.get("n"):
                    per.setdefault(n, {}).setdefault(b[:3], []).append((b, v["tok_s"]))
        for n in sorted(per, key=int):
            c = [x for _, x in per[n].get("ctl", [])]; m = [x for _, x in per[n].get("arm", [])]
            cells = " ".join(f"{b} {x:.0f}" for k in ("ctl", "arm") for b, x in per[n].get(k, []))
            d_ = f", arm vs ctl {100 * (statistics.mean(m) / statistics.mean(c) - 1):+.1f}%" if c and m else ""
            print(f"  {int(n) // 1024}K: {cells}{d_}")


def selftest():
    p = plan([64, 32], 2, "s", list(range(500)))
    assert [len(x[2]) for x in p] == [64, 64, 32, 32]
    assert len({tuple(x[2][:NONCE]) for x in p}) == 4  # every request starts differently
    w = [x[2][NONCE:] for x in p]
    assert w[0][0] == 0 and w[1][0] == 48 and w[2][0] == 96 and w[3][0] == 112  # distinct windows
    assert plan([64], 1, "s", list(range(500)))[0][2] == p[0][2]  # same seed, same prompt (paired boots)
    assert plan([600], 1, "s", list(range(500)))[0][2][NONCE + 500] == 0  # wraps on a short corpus
    s = summary([{"len": 8, "ttft_s": 1.0, "prompt_tokens": 2000}, {"len": 8, "ttft_s": 3.0, "prompt_tokens": 2000},
                 {"len": 9, "ttft_s": None, "prompt_tokens": None}])
    assert s["8"]["tok_s"] == 1000 and s["8"]["tok_s_each"] == [2000, 667] and s["9"]["n"] == 0
    print("cold selftest ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        selftest()
    elif sys.argv[1:2] == ["report"]:
        report(sys.argv[2:])
    elif sys.argv[1:2] == ["run"]:
        it = iter(sys.argv[2:]); a = {k.lstrip("-"): v for k, v in zip(it, it)}
        run(a)
    else:
        sys.exit(__doc__)
