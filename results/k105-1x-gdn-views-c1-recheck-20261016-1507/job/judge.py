#!/usr/bin/env python3
"""k105 judge: c1 decode recheck of GDN prefill views, VLLM_GDN_PREFILL_VIEWS=0 (ctl) vs 1 (arm), same image and cache.

Usage: judge.py <node dir> [<node dir>]   |   judge.py --selftest
Per Spark (node dir from the job-local Thunderdome, ctl-pK / arm-pK boots, pass K pairs ctl-pK with arm-pK):
  paired c1   fresh-c1 probe (T=0, same prompts in every boot), pairs (pass, prompt) pooled as thunderdome_report
              probe_cell: paired_decode_ab.compare tok_s (decode rate), step_ms (median step gap), tok/step; 95% bootstrap CI
  benchy c1   llama-benchy tg512 (c1) t/s, one value per boot, boot-level Welch 95% CI (t rounded up, df rounded down)
  acceptance  vllm:spec_decode_num_accepted_tokens_total / num_draft_tokens_total deltas around the benchy phase itself
  clocks      nvidia-smi 1 Hz during the boot's measurements (smi.csv): SM clock, temperature, power, event reasons
  facts       VLLM_GDN_PREFILL_VIEWS=0 on every ctl boot and 1 on every arm boot (env.txt); views log line on every arm
              boot and on no ctl boot (serve.log)
Rule per Spark: LOSS if the paired c1 tok_s CI or the benchy tg512 c1 CI lies wholly below -1%; else PASS when every
boot, pair count, benchy value, counter and fact is there; else EMPTY (reason printed). Last lines:
  SPARKS: <node>=PASS|LOSS|EMPTY ...   and   RULE=LOSS (any Spark LOSS) | EMPTY (any EMPTY) | PASS.
Env K105_NEED (default 6) = boots per side per Spark (lower only to replay older runs).
"""
import math, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from td_report_k105 import benchy, boots, marked, probe  # noqa: E402
from paired_decode_ab import compare  # noqa: E402

NEED = int(os.environ.get("K105_NEED", "6"))
FLOOR = -1.0
CELL = "tg512 (c1)"   # llama-benchy drops the " (c1)" suffix when c1 is the only concurrency: bare label accepted
VIEWS = "GDN prefill views: in-order"
T975 = (12.707, 4.303, 3.183, 2.777, 2.571, 2.447, 2.365, 2.307, 2.263, 2.229,
        2.201, 2.179, 2.161, 2.145, 2.132, 2.120, 2.110, 2.101, 2.094, 2.086)
TOT = re.compile(r"^vllm:spec_decode_num_(accepted|draft)_tokens_total(?:\{[^}]*\})? ([0-9.e+]+)")


def bval(p, cell):
    """t/s of a benchy cell in one boot, under its "(c1)" label or the bare one (single-concurrency runs)."""
    r = benchy(p)
    v = r.get(cell) or r.get(cell.replace(" (c1)", ""))
    return v[0] if v else None


def welch(c, a):
    """(delta %, lo %, hi %) of mean(a) vs mean(c), boot-level Welch 95% CI in % of mean(c)."""
    mc, ma = statistics.mean(c), statistics.mean(a)
    vc, va = statistics.variance(c) / len(c), statistics.variance(a) / len(a)
    se = math.sqrt(vc + va)
    df = (vc + va) ** 2 / (vc ** 2 / (len(c) - 1) + va ** 2 / (len(a) - 1)) if se else 1
    t = T975[min(len(T975), max(1, int(df))) - 1]
    d = ma - mc
    return d / mc * 100, (d - t * se) / mc * 100, (d + t * se) / mc * 100


def totals(text):
    out = {}
    for line in text.splitlines():
        m = TOT.match(line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def bench_acc(p):
    """(accepted, drafted) during the benchy phase of one boot, or None."""
    try:
        x, y = (totals(open(os.path.join(p, f"pos-benchy.{s}")).read()) for s in ("before", "after"))
    except OSError:
        return None
    if "draft" not in x or "draft" not in y or "accepted" not in y:
        return None
    return y["accepted"] - x.get("accepted", 0.0), y["draft"] - x["draft"]


def smi_rows(text):
    """nvidia-smi csv rows -> [(sm MHz, temp C, power W, event reasons hex)] (N/A fields skipped as None)."""
    out = []
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6:
            continue
        num = lambda s: float(s.split()[0]) if s and s[0].isdigit() else None
        out.append((num(f[1]), num(f[3]), num(f[4]), f[5]))
    return out


def smi(p):
    try:
        r = smi_rows(open(os.path.join(p, "smi.csv"), errors="replace").read())
    except OSError:
        return None
    sm = [x[0] for x in r if x[0]]
    if not sm:
        return None
    t = [x[1] for x in r if x[1] is not None]
    pw = [x[2] for x in r if x[2] is not None]
    ev = sum(1 for x in r if x[3] not in ("0x0000000000000000", "[N/A]", "N/A", ""))
    return {"n": len(r), "sm_med": statistics.median(sm), "sm_min": min(sm), "t_max": max(t) if t else None,
            "p_mean": statistics.mean(pw) if pw else None, "events": ev}


def facts(p):
    """(views env value, views log lines)."""
    try:
        env = re.findall(r"^VLLM_GDN_PREFILL_VIEWS=(\S*)", open(os.path.join(p, "env.txt")).read(), re.M)
    except OSError:
        env = []
    try:
        n = open(os.path.join(p, "serve.log"), errors="replace").read().count(VIEWS)
    except OSError:
        n = None
    return (env[0] if env else None), n


def decide(paired_hi, benchy_hi, missing):
    """LOSS beats everything: a wholly negative CI is a result even when another measurement is missing."""
    if (paired_hi is not None and paired_hi < FLOOR) or (benchy_hi is not None and benchy_hi < FLOOR):
        return "LOSS"
    return "EMPTY" if missing else "PASS"


def node(d):
    name, out, missing = os.path.basename(d.rstrip("/")), [], []
    C, A = boots(d, "ctl"), boots(d, "arm")
    for m in ("FAILED", "COLD", "DIED"):
        for b in marked(d, m):
            missing.append(f"{b}: {m}")
    if len(C) < NEED or len(A) < NEED:
        missing.append(f"boots up: ctl {len(C)} arm {len(A)} (need {NEED} each)")
    # paired c1 (fresh-c1, pooled over passes)
    a, b = {}, {}
    for k in sorted(set(C) & set(A)):
        x, y = probe(C[k], "fresh-c1"), probe(A[k], "fresh-c1")
        if x and y:
            a.update({(k, i): v for i, v in x.items()})
            b.update({(k, i): v for i, v in y.items()})
    res = compare(a, b) if a else {}
    t = res.get("tok_s")
    out.append(f"== {name}")
    for key in ("tok_s", "step_ms", "tok_per_step"):
        r = res.get(key)
        out.append(f"  paired c1 {key:12s} " + (f"n={r['n']:3d}  {r['mean_pct']:+6.2f}%  CI95 [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  arm better {r['b_better']}/{r['n']}" if r else "missing"))
    if not t or t["n"] < 4 * NEED:
        missing.append(f"paired c1 pairs {t['n'] if t else 0} (need {4 * NEED})")
    # benchy tg512 c1, one value per boot
    c = [x for x in (bval(p, CELL) for _, p in sorted(C.items())) if x]
    v = [x for x in (bval(p, CELL) for _, p in sorted(A.items())) if x]
    w = welch(c, v) if len(c) >= 2 and len(v) >= 2 else None
    out.append(f"  benchy {CELL}  " + (f"{w[0]:+6.2f}%  CI [{w[1]:+.2f}, {w[2]:+.2f}]  control {' '.join(f'{x:.1f}' for x in c)} | arm {' '.join(f'{x:.1f}' for x in v)}" if w else f"too few boots ({len(c)}/{len(v)})"))
    if len(c) < NEED or len(v) < NEED:
        missing.append(f"benchy {CELL}: {len(c)}/{len(v)} boots (need {NEED})")
    pp = [[x for x in (bval(p, "pp2048 (c1)") for p in B.values()) if x] for B in (C, A)]
    if all(len(x) >= 2 for x in pp):
        pw = welch(*pp)
        out.append(f"  benchy pp2048 (c1) {pw[0]:+6.2f}%  CI [{pw[1]:+.2f}, {pw[2]:+.2f}]  (reported, not judged)")
    # acceptance during benchy
    for who, B in (("ctl", C), ("arm", A)):
        per = {k: bench_acc(p) for k, p in sorted(B.items())}
        got = [x for x in per.values() if x and x[1] > 0]
        if len(got) < len(B) or not got:
            missing.append(f"{who}: benchy acceptance counters on {len(got)}/{len(B)} boots")
        rate = sum(x[0] for x in got) / sum(x[1] for x in got) if got else None
        out.append(f"  benchy acceptance {who}: " + (f"{rate:.4f} pooled; per boot " + " ".join(f"p{k}:{x[0] / x[1]:.3f}" if x and x[1] else f"p{k}:-" for k, x in per.items()) if rate is not None else "missing"))
    # clocks and facts per boot
    for who, B, want in (("ctl", C, "0"), ("arm", A, "1")):
        for k, p in sorted(B.items()):
            s, (env, n) = smi(p), facts(p)
            out.append(f"  {who}-p{k}: views env {env} log lines {n}; " + (
                f"SM MHz median {s['sm_med']:.0f} min {s['sm_min']:.0f}, temp max {s['t_max']}, power mean {s['p_mean']:.1f} W, "
                f"event-reason samples {s['events']}/{s['n']}" if s and s["p_mean"] is not None else "smi missing"))
            if env != want:
                missing.append(f"{who}-p{k}: VLLM_GDN_PREFILL_VIEWS={env} (want {want})")
            if n is None or (want == "1" and n < 1) or (want == "0" and n > 0):
                missing.append(f"{who}-p{k}: views log lines {n} (want {'>= 1' if want == '1' else '0'})")
            if not s:
                missing.append(f"{who}-p{k}: no nvidia-smi samples")
    v_ = decide(t["ci95"][1] if t else None, w[2] if w else None, missing)
    for x in missing:
        out.append(f"  missing: {x}")
    out.append(f"  {name}: {v_}")
    return v_, out


def main(dirs):
    res = []
    for d in dirs:
        v, lines = node(d)
        print("\n".join(lines))
        res.append((os.path.basename(d.rstrip("/")), v))
    vs = [v for _, v in res]
    print("SPARKS: " + " ".join(f"{n}={v}" for n, v in res))
    print("RULE=" + ("LOSS" if "LOSS" in vs else "EMPTY" if "EMPTY" in vs or not vs else "PASS"))


def selftest():
    # k101 dgx-01 tg512 c1 (inside the old 6.27% band): the boot CI is wholly below -1%
    d, lo, hi = welch([59.5, 57.5, 59.3, 61.2], [54.6, 57.0, 57.1, 54.8])
    assert -6.0 < d < -5.8 and hi < -1.4, (d, lo, hi)
    assert decide(None, hi, []) == "LOSS" and decide(None, hi, ["x"]) == "LOSS"
    assert decide(-1.2, 0.5, []) == "LOSS"                                   # paired c1 CI wholly below -1%
    assert decide(-0.5, welch([100.0] * 6, [99.5] * 6)[2], []) == "PASS"    # sub-1% shift
    assert decide(0.3, welch([90.0, 110.0] * 3, [95.0] * 6)[2], []) == "PASS"  # uncertain shift
    assert decide(0.3, 0.5, ["boots"]) == "EMPTY" and decide(None, None, []) == "PASS"
    m = ('vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 3484.0\n'
         'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 2210.0\n'
         'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 725.0\n')
    assert totals(m) == {"draft": 3484.0, "accepted": 2210.0}, totals(m)
    r = smi_rows("2026/10/16 11:37:55.254, 2171 MHz, [N/A], 56, 21.96 W, 0x0000000000000000\n")
    assert r == [(2171.0, 56.0, 21.96, "0x0000000000000000")], r
    print("judge selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif sys.argv[1:]:
        main(sys.argv[1:])
    else:
        sys.exit(__doc__)
