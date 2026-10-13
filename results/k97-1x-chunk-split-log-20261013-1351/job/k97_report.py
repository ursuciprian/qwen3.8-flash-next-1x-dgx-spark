#!/usr/bin/env python3
"""k97 report (stdlib only): prefill chunk lists, split reasons and memory per max_num_batched_tokens budget.

Inputs in RES: marks.tsv ("budget<TAB>phase<TAB>start_epoch<TAB>end_epoch<TAB>rc", phases boot/prebake/prompts/
capture), guard-dgx01.log ("<epoch> <MemAvailable MiB>" per second), memtotal.txt (MiB), b<budget>/k97.log (the
K97SPLIT / K97STEP lines of that boot's serve log).
Usage: k97_report.py report RES | k97_report.py selftest
"""
import os
import re
import sys

STEP = re.compile(r"K97STEP step=(\d+) req=(\S+) prompt=(\d+) start=(\d+) n=(\d+) budget=(\d+)")
SPLIT = re.compile(r"K97SPLIT req=(\S+) prompt=(\d+) start=(\d+) asked=(\d+) aligned_end=(\d+) block=(\d+) "
                   r"internal_ckpt=(\S+) eagle=(\S+) stops=(\[.*?\]) final=(-?\d+)")
STOP = re.compile(r"\((\d+), '(\w+)'\)")


def parse(lines):
    """-> {req: {"prompt": n, "chunks": [(start, n, reason)], "block": b}}"""
    steps, splits = {}, {}
    for line in lines:
        m = STEP.search(line)
        if m:
            _, req, prompt, start, n, budget = m.groups()
            steps.setdefault(req, {"prompt": int(prompt), "chunks": {}, "budget": int(budget)})["chunks"][int(start)] = int(n)
            continue
        m = SPLIT.search(line)
        if m:
            req, prompt, start, asked, aligned, block, _, _, stops, final = m.groups()
            splits[(req, int(start))] = (int(prompt), int(asked), int(aligned), int(block),
                                         [(int(p), w) for p, w in STOP.findall(stops)], int(final))
    out = {}
    for req, s in steps.items():
        chunks, block = [], None
        for start in sorted(s["chunks"]):
            n = s["chunks"][start]
            sp = splits.get((req, start))
            if sp is None:
                why = "no split line"
            else:
                prompt, asked, aligned, block, stops, final = sp
                if stops and stops[0][0] - start == final:
                    why = f"stop {stops[0][1]} @{stops[0][0]}"
                elif aligned < start + asked:
                    why = f"block align @{aligned}"
                elif start + asked < prompt:
                    budget = s["budget"]
                    why = "budget" if asked >= budget else "clipped before split"
                else:
                    why = "prompt end"
                if n != final:
                    why += f" (then clipped {final} -> {n})"
            chunks.append((start, n, why))
        out[req] = {"prompt": s["prompt"], "chunks": chunks, "block": block}
    return out


def minmem(guard, a, b):
    m = None
    for line in guard:
        p = line.split()
        if len(p) == 2 and p[1].isdigit():
            t = int(p[0])
            if a <= t <= b and (m is None or int(p[1]) < m):
                m = int(p[1])
    return m


def report(res):
    L, verdict = [], []
    marks = [l.rstrip("\n").split("\t") for l in open(os.path.join(res, "marks.tsv"))] if os.path.exists(
        os.path.join(res, "marks.tsv")) else []
    guard = open(os.path.join(res, "guard-dgx01.log")).read().splitlines() if os.path.exists(
        os.path.join(res, "guard-dgx01.log")) else []
    total = int(open(os.path.join(res, "memtotal.txt")).read().split()[0]) if os.path.exists(
        os.path.join(res, "memtotal.txt")) else None
    budgets = sorted({int(m[0]) for m in marks if m and m[0].isdigit()})
    L.append("== memory: min MemAvailable (GiB) and peak used (MemTotal - min available) per phase; guard aborts below 2 GiB")
    L.append("budget  phase      rc   seconds  min_avail_GiB  peak_used_GiB")
    for m in marks:
        if len(m) < 5:
            continue
        b, ph, a, e, rc = m[:5]
        mm = minmem(guard, int(float(a)), int(float(e)))
        used = "n/a" if mm is None or total is None else f"{(total - mm) / 1024:.1f}"
        L.append(f"{b:>6}  {ph:<9} {rc:>3} {int(float(e) - float(a)):>8}  "
                 f"{'n/a' if mm is None else f'{mm / 1024:.1f}':>13}  {used:>13}")
    for b in budgets:
        p = os.path.join(res, f"b{b}", "k97.log")
        L.append("")
        L.append(f"== budget {b}: prefill chunks per cold prompt (start, tokens, why the chunk ends there)")
        if not os.path.exists(p):
            L.append("  no K97 log (boot failed?)")
            verdict.append(f"b{b}=noboot")
            continue
        reqs = parse(open(p, errors="replace").read().splitlines())
        if not reqs:
            L.append("  no K97STEP lines")
            verdict.append(f"b{b}=nolog")
            continue
        for req, r in sorted(reqs.items(), key=lambda kv: (kv[1]["prompt"], kv[0])):
            ns = [n for _, n, _ in r["chunks"]]
            ok = sum(ns) == r["prompt"]
            L.append(f"  prompt {r['prompt']} tokens: {len(ns)} steps {ns}{'' if ok else ' (sum != prompt: partial log)'}"
                     f"  block {r['block']}")
            for start, n, why in r["chunks"]:
                L.append(f"    {start:>7} +{n:>6}  {why}")
        short = "; ".join(f"{r['prompt'] // 1024}K={len(r['chunks'])}x{[n for _, n, _ in r['chunks']] if len(r['chunks']) <= 6 else ''}"
                          for r in sorted(reqs.values(), key=lambda r: r["prompt"]))
        verdict.append(f"b{b}: {short}".replace(" ", ""))
    L.append("")
    L.append("VERDICT " + " | ".join(verdict))
    return "\n".join(L)


def selftest():
    import tempfile
    log = [
        "K97SPLIT req=a prompt=16384 start=0 asked=9072 aligned_end=9072 block=3024 internal_ckpt=False eagle=True stops=[] final=9072",
        "K97STEP step=1 req=a prompt=16384 start=0 n=9072 budget=9072",
        "K97SPLIT req=a prompt=16384 start=9072 asked=7312 aligned_end=16384 block=3024 internal_ckpt=False eagle=True stops=[(12096, 'last_cache'), (15120, 'reuse')] final=3024",
        "K97STEP step=2 req=a prompt=16384 start=9072 n=3024 budget=9072",
        "K97SPLIT req=a prompt=16384 start=12096 asked=4288 aligned_end=16384 block=3024 internal_ckpt=False eagle=True stops=[(15120, 'reuse')] final=3024",
        "K97STEP step=3 req=a prompt=16384 start=12096 n=3024 budget=9072",
        "K97SPLIT req=a prompt=16384 start=15120 asked=1264 aligned_end=16384 block=3024 internal_ckpt=False eagle=True stops=[] final=1264",
        "K97STEP step=4 req=a prompt=16384 start=15120 n=1264 budget=9072",
    ]
    clip = ["K97SPLIT req=c prompt=16384 start=0 asked=5000 aligned_end=5000 block=16 internal_ckpt=False eagle=True stops=[] final=5000",
            "K97STEP step=9 req=c prompt=16384 start=0 n=5000 budget=9072"]
    assert parse(clip)["c"]["chunks"][0][2] == "clipped before split", parse(clip)
    r = parse(log)["a"]
    assert [n for _, n, _ in r["chunks"]] == [9072, 3024, 3024, 1264], r
    assert r["chunks"][0][2] == "budget" and r["chunks"][1][2] == "stop last_cache @12096", r
    assert r["chunks"][2][2] == "stop reuse @15120" and r["chunks"][3][2] == "prompt end", r
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/b9072")
        open(f"{d}/b9072/k97.log", "w").write("\n".join(log))
        open(f"{d}/marks.tsv", "w").write("9072\tboot\t100\t110\t0\n9072\tprompts\t110\t120\t0\n12096\tboot\t130\t140\t1\n")
        open(f"{d}/guard-dgx01.log", "w").write("\n".join(f"{t} {9000 - t}" for t in range(100, 141)))
        open(f"{d}/memtotal.txt", "w").write("122000\n")
        out = report(d)
        assert "16K=4x[9072,3024,3024,1264]" in out and "b12096=noboot" in out, out
        assert "  9072  boot        0       10            8.7          110.5" in out, out  # 8890 MiB at t=110
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        selftest()
    elif sys.argv[1:2] == ["report"]:
        print(report(sys.argv[2]))
    else:
        sys.exit(__doc__)
