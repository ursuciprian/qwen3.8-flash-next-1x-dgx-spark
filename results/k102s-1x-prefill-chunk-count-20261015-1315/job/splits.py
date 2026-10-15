#!/usr/bin/env python3
"""k102: prefill chunk lists from the VLLM_DEBUG_PREFILL_SPLITS lines of a serve log (stdlib).

  splits.py table <serve.log>              per prompt: steps, chunk list, why each chunk ends there
  splits.py steps <serve.log> <prompt>     the chunk list of the last prompt of that length, as "n1,n2,..." ("" if none)
  splits.py selftest

One line per split decision ("prefill split req=R prompt=P start=S requested=Q scheduled=N end=E reason=W {...}").
A request asked again at the same start (not admitted that step) logs again; the last line per (request, start)
wins. A chunk list whose starts do not chain (start + scheduled != next start) is flagged: something after the
split (lookahead reserve, encoder budget) cut the step, and the list is not the schedule.
"""
import re
import sys

LINE = re.compile(r"prefill split req=(\S+) prompt=(\d+) start=(\d+) requested=(\d+) scheduled=(\d+) end=(\d+) reason=(\S+)")


def parse(text):
    """-> [(req, prompt, [(start, n, reason)], chained)] in first-seen order"""
    reqs, order = {}, []
    for m in LINE.finditer(text):
        req, prompt, start, _, n, _, why = m.groups()
        if req not in reqs:
            reqs[req] = {"prompt": int(prompt), "by_start": {}}
            order.append(req)
        reqs[req]["by_start"][int(start)] = (int(n), why)
    out = []
    for req in order:
        r = reqs[req]
        chunks = [(s, n, w) for s, (n, w) in sorted(r["by_start"].items())]
        chained = chunks[0][0] == 0 and all(a[0] + a[1] == b[0] for a, b in zip(chunks, chunks[1:])) \
            and chunks[-1][0] + chunks[-1][1] == r["prompt"]
        out.append((req, r["prompt"], chunks, chained))
    return out


def table(text):
    lines = []
    for _, prompt, chunks, chained in parse(text):
        lines.append(f"  prompt {prompt}: {len(chunks)} steps [{', '.join(str(n) for _, n, _ in chunks)}]"
                     f"{'' if chained else '  NOT CHAINED (cut after the split)'}")
        lines += [f"    {s:>7} + {n:>6}  {w}" for s, n, w in chunks]
    return "\n".join(lines)


def steps(text, prompt):
    hit = [c for _, p, c, ok in parse(text) if p == prompt and ok]
    return ",".join(str(n) for _, n, _ in hit[-1]) if hit else ""


def selftest():
    log = "\n".join([
        "INFO prefill split req=a prompt=16384 start=0 requested=9072 scheduled=9072 end=9072 reason=budget {}",
        "INFO prefill split req=b prompt=19 start=0 requested=19 scheduled=19 end=19 reason=prompt-end {}",
        "INFO prefill split req=a prompt=16384 start=9072 requested=7312 scheduled=3024 end=12096 reason=reachable {}",
        "INFO prefill split req=a prompt=16384 start=12096 requested=4288 scheduled=3024 end=15120 reason=last-cache-position {}",
        "INFO prefill split req=a prompt=16384 start=15120 requested=1264 scheduled=1264 end=16384 reason=prompt-end {}",
        "INFO prefill split req=c prompt=16384 start=0 requested=8192 scheduled=6048 end=6048 reason=block-align {}",
        "INFO prefill split req=c prompt=16384 start=6000 requested=8192 scheduled=6048 end=12048 reason=block-align {}",
    ])
    assert steps(log, 16384) == "9072,3024,3024,1264", steps(log, 16384)
    assert steps(log, 19) == "19"
    t = table(log)
    assert "prompt 16384: 4 steps [9072, 3024, 3024, 1264]" in t and "NOT CHAINED" in t, t
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["table"] and len(a) == 2:
        print(table(open(a[1], errors="replace").read()))
    elif a[:1] == ["steps"] and len(a) == 3:
        print(steps(open(a[1], errors="replace").read(), int(a[2])))
    else:
        sys.exit(__doc__)
