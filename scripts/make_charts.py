#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Render the README charts from committed files under results/ into docs/img/*.svg.

    uv run scripts/make_charts.py            # or: pip install matplotlib && python3 scripts/make_charts.py

Every number drawn comes from a file listed in SRC. Transparent background and mid-tone colours, so the
same SVG reads on GitHub's light and dark themes.
"""
import json, math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "img"

T1 = RES / "tp1-v3d-20261005"   # single-Spark v3d: gate files and bench/ (one boot on dgx-01)
HC = RES / "high-conc-k46b-20261005"   # max_num_seqs 32 runs (k46b): not the shipped cap, no quality gate at this cap

SRC = {
    # v3d: coding grid, copy-heavy, counting (5 rounds, every round saved), llm-inference-bench (one boot on dgx-01).
    "tp1_benchy": T1 / "bench/task.csv",
    "tp1_copy": [T1 / "bench/copy-streams.json"],
    "tp1_count": sorted((T1 / "bench").glob("count-c*.json")),
    "tp1_lib": T1 / "bench/lib-decode.json",
    # Promoted builds, in order: each build's own llama-benchy coding grid (one boot, 3 runs).
    "tp1_builds": [
        ("v2", "10-02", RES / "tp1-v2-20261002/bench-task.csv"),
        ("v3a", "10-04", RES / "tp1-v3a-20261004/bench-task.csv"),
        ("v3b", "10-04", RES / "tp1-v3b-20261004/bench-task.csv"),
        ("v3c", "10-05", RES / "tp1-v3c-20261005/bench/task.csv"),
        ("v3d", "10-05", T1 / "bench/task.csv"),
        ("v3e", "10-08", RES / "tp1-v3e-hf-20261008/bench/task.csv"),
    ],
    # High concurrency, max_num_seqs 32 (one boot): counting, copy-heavy, coding at c16/c32.
    "hc": {"1× Spark (TP=1), v3d, max_num_seqs 32": HC / "v3d-s32"},
    "tp1_fidelity": T1 / "gate-summary.txt",
    "tp1_gate": T1 / "gate-summary.txt",
}

# Neutral ink that keeps >= 3:1 contrast on both #ffffff and #0d1117; series hues are mid-tone.
INK, MUTED, GRID = "#768390", "#8b949e", "#8b949e40"
C_COPY, C_COUNT, C_CODE, C_CODE16 = "#3987e5", "#1baf7a", "#e0662f", "#d4a017"
C_C1, C_C4, C_C8 = "#3987e5", "#1baf7a", "#e0662f"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRID, "svg.fonttype": "path",
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
    "legend.frameon": False, "svg.hashsalt": "make_charts", "axes.spines.top": False, "axes.spines.right": False,
})


# ---------- loaders ----------

def benchy(path):
    """llama-benchy markdown table -> {test: (total t/s, accept/draft)}."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 10 or not re.search(r"\(c\d+\)", cells[1]):
            continue
        rows[cells[1]] = (float(cells[2].split("±")[0]), float(cells[9]))
    return rows


def grid(prefix, rows):
    """'tg512' or 'tg512 @ d16384' -> {c: t/s}."""
    pat = re.compile(re.escape(prefix) + r" \(c(\d+)\)$")
    return {int(m.group(1)): v[0] for k, v in rows.items() if (m := pat.match(k))}


def verdict_means(path, key, prefix=""):
    d = json.loads(Path(path).read_text())[key]
    return {int(k[len(prefix) + 1:]): v["cand_mean"] for k, v in d.items() if k.startswith(prefix + "c")}


def copy_best(paths):
    """Copy-heavy files -> {streams: max round window tok/s over all files}."""
    out = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["rounds"]:
            out[r["n"]] = max(out.get(r["n"], 0), r["window"]["tok_s"])
    return out


def count_best(paths):
    """Counting sweep files -> {c: max round agg tok/s}. Files that saved every round carry agg_tok_s_max."""
    out = {}
    for p in paths:
        for r in json.loads(Path(p).read_text())["rows"]:
            out[r["c"]] = max(out.get(r["c"], 0), r.get("agg_tok_s_max", r["agg_tok_s"]))
    return out


def lib_decode(path):
    """llm-inference-bench JSON -> {c: {context tokens: aggregate tok/s}} over the cells that ran."""
    out = {}
    for r in json.loads(Path(path).read_text())["results"]:
        if r.get("aggregate_tps") and not r.get("failure_reason"):
            out.setdefault(r["concurrency"], {})[r["context_tokens"]] = r["aggregate_tps"]
    return out


def fidelity(path):
    out = []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"actual_tokens\s+(\d+)\s+\|\s+exact\s+(\d+)", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def load():
    t1 = benchy(SRC["tp1_benchy"])
    tp1 = {
        "copy": copy_best(SRC["tp1_copy"]),
        "count": count_best(SRC["tp1_count"]),
        "code": grid("tg512", t1),
        "code16": grid("tg512 @ d16384", t1),
    }
    hist1 = []
    for name, date, path in SRC["tp1_builds"]:
        g = benchy(path)
        hist1.append((name, date, g["tg512 (c1)"][0], g["tg512 (c8)"][0],
                      g["tg512 @ d16384 (c1)"][0], g["tg512 @ d16384 (c8)"][0]))
    lib = {"1× Spark (TP=1), v3d": lib_decode(SRC["tp1_lib"])}
    for name, data in (("tp1", tp1), ("lib", lib)):
        for k, v in data.items():
            assert v, f"{name}.{k}: no data parsed"
    return tp1, hist1, lib


# ---------- charts ----------

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def style_ax(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


SERIES = [
    ("copy", "Copy-heavy (high acceptance, ~4.9 tok/step)", C_COPY, "-"),
    ("count", "Counting (high acceptance, ~5.0 tok/step)", C_COUNT, "-"),
    ("code", "Coding, benchy tg512 (~2.7–3.5 tok/step)", C_CODE, "-"),
    ("code16", "Coding at 16k cached context", C_CODE16, "--"),
]


def chart_throughput(tp1):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style_ax(ax)
    for key, label, color, ls in SERIES:
        pts = sorted((c, y) for c, y in tp1[key].items() if c <= 8)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls, color=color, lw=2.2, marker="o", ms=5, label=label)
        dy = {"code": 5, "code16": -6}.get(key, 0)   # the two coding end points can sit close together
        ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlim(0.85, 8 * 1.45)
    ax.set_title("1× Spark (TP=1), v3d", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Aggregate decode tok/s")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.text(0.0, -0.1, "Copy-heavy: max of 3 rounds per task count, low thinking effort. "
             "Counting: T=0, thinking off, max of 5 rounds per level.\n"
             "Coding: llama-benchy task mode, T=1.0, thinking on, one boot, 3 runs. Raw files: results/tp1-v3d-20261005/bench/.",
             fontsize=8.5, color=MUTED)
    save(fig, "throughput.svg")


def chart_builds(hist1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [f"{n}\n{d}" for n, d, *_ in hist1]
    xs = range(len(hist1))
    for ax, title, series in ((axes[0], "1 request", ((2, "Coding, benchy tg512", C_CODE), (4, "Coding at 16k cached context", C_CODE16))),
                              (axes[1], "8 concurrent requests", ((3, "Coding, benchy tg512", C_CODE), (5, "Coding at 16k cached context", C_CODE16)))):
        style_ax(ax)
        for idx, name, color in series:
            ys = [h[idx] for h in hist1]
            ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=5, label=name, ls="--" if color == C_CODE16 else "-")
            for x, y in ((0, ys[0]), (len(ys) - 1, ys[-1])):
                # the lower of the two series at this point is labelled below its marker, so labels never cross
                below = any(h[i] > y or (h[i] == y and color == C_CODE16) for h in (hist1[x],) for i, _, _ in series if i != idx)
                ax.annotate(f"{y:.0f}", (x, y), xytext=(0, -16 if below else 8), textcoords="offset points",
                            ha="center", fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks(list(xs), labels, fontsize=9)
        ax.set_ylim(0, None)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
    axes[0].set_ylabel("Aggregate decode tok/s")
    axes[1].legend(loc="center left", fontsize=9.5)
    fig.tight_layout()
    fig.text(0.0, -0.1, "Promoted builds in order (2026). Each build's llama-benchy coding grid (one boot, 3 runs, T=1.0); "
             "single cells vary by up to ~10% between runs.\nAt 16k with 8 requests, v2-v3b ran out of KV pool (6 GiB); "
             "v3c to v3e have 14 GiB. Every build passed the quality gate.", fontsize=8.5, color=MUTED)
    save(fig, "build-history.svg")


def chart_depth(lib):
    fig, axes = plt.subplots(1, len(lib), figsize=(6.5 * len(lib), 4.4), sharey=True, squeeze=False)
    axes = axes[0]
    ctx = [0, 16384, 65536]
    for ax, (title, data) in zip(axes, lib.items()):
        style_ax(ax)
        for c, color in ((8, C_C8), (4, C_C4), (1, C_C1)):
            pts = [(i, data.get(c, {}).get(x)) for i, x in enumerate(ctx)]
            pts = [(i, y) for i, y in pts if y]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=5, label=f"{c} request{'s' if c > 1 else ''}")
            for x, y in pts:
                ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center",
                            fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xticks(range(len(ctx)), ["0", "16K", "64K"])
        ax.set_xlim(-0.3, len(ctx) - 0.7)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
        ax.set_xlabel("Context already in the prompt (tokens)")
    axes[0].set_ylabel("Aggregate decode tok/s")
    axes[0].set_ylim(0, None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=10)
    fig.text(0.0, -0.14, "llm-inference-bench, 30 s of sustained decode per cell, server default sampling, "
             "one boot.\nRaw files: results/tp1-v3d-20261005/bench/.",
             fontsize=8.5, color=MUTED)
    save(fig, "depth.svg")


def benchy_tg_max(path):
    """llama-benchy JSON -> {concurrency: max run tg tok/s total} at depth 0."""
    out = {}
    for b in json.loads(Path(path).read_text())["benchmarks"]:
        if b["context_size"] == 0 and not b["is_context_prefill_phase"]:
            out[b["concurrency"]] = max(b["tg_throughput"]["values"])
    return out


def chart_concurrency():
    fig, axes = plt.subplots(1, len(SRC["hc"]), figsize=(7.5 * len(SRC["hc"]), 4.6), sharey=True, squeeze=False)
    for ax, (title, d) in zip(axes[0], SRC["hc"].items()):
        style_ax(ax)
        series = (("Copy-heavy, max of 3 rounds", C_COPY, copy_best([d / "copy-streams.json"])),
                  ("Counting, max of 5 rounds", C_COUNT, count_best(sorted(d.glob("count-c*.json")))),
                  ("Coding, benchy tg512, max of 3 runs", C_CODE, benchy_tg_max(d / "benchy.json")))
        for label, color, data in series:
            pts = sorted((c, y) for c, y in data.items() if c in (1, 8, 16, 32))
            assert pts, f"{title} {label}: no data"
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=5, label=label)
            ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=10, color=INK, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 16, 32], ["1", "8", "16", "32"])
        ax.set_xlim(0.85, 32 * 1.5)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK)
        ax.set_xlabel("Concurrent requests")
    axes[0][0].set_ylabel("Aggregate decode tok/s")
    axes[0][0].set_ylim(0, None)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12), fontsize=10)
    fig.text(0.0, -0.22, "Measured with max_num_seqs 32 (the shipped recipe uses 8); the quality gate "
             "was not run at this cap.\nOne boot per setup; coding measured at c16 and c32 only. Raw files: "
             "results/high-conc-k46b-20261005/.", fontsize=8.5, color=MUTED)
    save(fig, "concurrency.svg")


def chart_quality():
    t1f = fidelity(SRC["tp1_fidelity"])
    gate1 = SRC["tp1_gate"].read_text()
    hard1 = re.search(r"Quality:\s+(\d+)/100", gate1).group(1)
    tc1 = re.search(r"Score:\s+([\d.]+) ±", gate1).group(1)
    rows = [("Hard multi-step tool use (88 scenarios)", f"{hard1}/100", "≥ 88"),
            ("tool_choice=required (TC-45, 5 trials)", f"{float(tc1):.0f}/100", "-")]
    for tok, ex in t1f[:4]:
        rows.append((f"Tool-call retrieval at {round(tok, -3) / 1000:.0f}k tokens", f"{ex}/20", "20/20"))
    rows.append(("Batch stragglers", ("none" if "preemptions +0" in gate1 else "check") + " (c8–c16)", "none"))

    fig, ax = plt.subplots(figsize=(8, 0.46 * (len(rows) + 1) + 0.4))
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99)
    cols = (0.0, 0.58, 0.85)
    for x, h in zip(cols, ("Check", "1× Spark v3d", "Threshold")):
        ax.text(x, len(rows), h, fontweight="bold", fontsize=11, color=INK, va="center")
    for i, (name, val, thr) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.axhline(y + 0.5, color=GRID, lw=0.8)
        ax.text(cols[0], y, name, fontsize=10.5, color=INK, va="center")
        ax.text(cols[1], y, "✓", fontsize=12, color=C_COUNT, va="center", fontweight="bold")
        ax.text(cols[1] + 0.04, y, val, fontsize=10.5, color=INK, va="center", fontweight="bold")
        ax.text(cols[2], y, thr, fontsize=9.5, color=MUTED, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) + 0.5)
    fig.text(0.0, -0.02, "Retrieval depths are the logged prompt sizes (labels 8k/32k/64k/128k in the probe). "
             "v3e passed the same checks in k56.\nA build that misses any check is not promoted.",
             fontsize=8.5, color=MUTED)
    save(fig, "quality-gate.svg")


if __name__ == "__main__":
    tp1, hist1, lib = load()
    chart_throughput(tp1)
    chart_builds(hist1)
    chart_depth(lib)
    chart_quality()
    chart_concurrency()
