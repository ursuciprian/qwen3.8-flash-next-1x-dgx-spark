#!/usr/bin/env python3
"""Cold TTFT verdict per Thunderdome node dir (k88 inline rule as a file, k90).

  cold_judge.py RES <node dir name>...   |   cold_judge.py --selftest
Per node and prompt length: arm vs control cold tok/s (mean over boots, from <boot>/hook/cold.json); noise = the
control's boot-to-boot spread (at least 1%; 3% with one control boot). A node passes when no length is worse than
noise and 16K is better than noise. Last line COLD_ALL=PASS|FAIL.
"""
import glob, json, os, statistics, sys


def verdict(per):
    """per: {len: {"ctl": [tok/s], "arm": [tok/s]}} -> (lines, ok)"""
    out, ok = [], bool(per)
    for n in sorted(per, key=int):
        c, m = per[n].get("ctl", []), per[n].get("arm", [])
        if not c or not m:
            out.append(f"{int(n)//1024}K: missing boots"); ok = False; continue
        d = 100 * (statistics.mean(m) / statistics.mean(c) - 1)
        noise = max(1.0, 100 * (max(c) - min(c)) / statistics.mean(c)) if len(c) > 1 else 3.0
        tag = "better" if d > noise else "WORSE" if d < -noise else "flat"
        if tag == "WORSE" or (n == "16384" and tag != "better"):
            ok = False
        out.append(f"{int(n)//1024:>4}K: control {statistics.mean(c):6.0f} arm {statistics.mean(m):6.0f} tok/s ({d:+.1f}%, noise {noise:.1f}%) {tag}")
    return out, ok


def main(res, nodes):
    ok_all = True
    for a in nodes:
        per = {}
        for b in sorted(glob.glob(f"{res}/{a}/*-p[0-9]")):
            try:
                s = json.load(open(f"{b}/hook/cold.json"))["summary"]
            except (OSError, ValueError, KeyError):
                continue
            for n, v in s.items():
                if v.get("n"):
                    per.setdefault(n, {}).setdefault(os.path.basename(b)[:3], []).append(v["tok_s"])
        lines, ok = verdict(per)
        for l in lines:
            print(f"{a} {l}")
        print(f"{a} COLD={'PASS' if ok else 'FAIL'}")
        ok_all &= ok
    print(f"COLD_ALL={'PASS' if ok_all else 'FAIL'}")


def selftest():
    good = {"16384": {"ctl": [2000, 2010], "arm": [2100, 2110]}, "65536": {"ctl": [1900, 1905], "arm": [1902, 1899]}}
    assert verdict(good)[1]
    flat16 = {"16384": {"ctl": [2000, 2010], "arm": [2005, 2008]}}
    assert not verdict(flat16)[1]
    worse = dict(good, **{"131072": {"ctl": [1800, 1802], "arm": [1700, 1705]}})
    assert not verdict(worse)[1]
    assert not verdict({})[1]
    print("cold_judge selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2:])
    else:
        sys.exit(__doc__)
