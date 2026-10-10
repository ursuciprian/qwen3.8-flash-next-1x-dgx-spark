"""Jev ship/no-ship on k56c + k56d pooled (drafter D3 = refit run3a vs shipped D1 on the 1x v2.1.0 stack). Gate in code first."""
import json, subprocess, sys, time
from typesafe_sdk import Choice, TypeSafeClient

RULES = (
    "Quality gate (non-negotiable, checked in code): hardmode >= 88, TC-45 100, fidelity 20/20 exact at "
    "8k/32k/64k/128k plus 2 extra 128k seeds, stragglers c8-c16 with 0 preemptions. Speed: ship if the arm is faster "
    "beyond the cell's noise band in at least one cell and no cell is worse beyond noise on either Spark; a beyond-noise "
    "loss anywhere caps the verdict at ship_with_caveat at best, a low-concurrency (c1-c4) loss that is not tiny means "
    "reject. A cell whose difference is inside its noise band is neither a win nor a loss. The arm only changes the MTP "
    "drafter weights (same image, same main weights), so an acceptance rise per draft position is the intended effect."
)
cell = lambda d, n: {"pct": d, "noise_pct": n, "beyond_noise": abs(d) > n}
facts = {
    "arm": "drafter D3 (refit run 3a: run 1 init, 10.66M anchors, one epoch, lr decay to 0.1x); offline T=0 +2.21 pt/pos over D1",
    "baseline": "shipped 1x v2.1.0 drafter D1 (refit run 1)",
    "design": "Thunderdome per Spark, k56c (2 boots per side, full cell set) + k56d (4 more boots per side on the cells k56c left in doubt: fresh-c8, pp2048 c1, tg512 c1/c8, benchy 5 runs per boot), pooled to 6 boots per side",
    "quality_gate": {"pass": True, "hardmode": [93], "tc45": [100],
                      "fidelity": "20/20 exact at 8k/32k/64k/128k and two extra 128k seeds 20/20",
                      "stragglers_preemptions": 0, "min_mem_available_gib": 13.89},
    "post_checks": "every boot: 0 measured b12x plans, KV pool 993,754 tokens on both sides, AOT loaded, no Traceback, MemAvailable >= 6 GiB",
    "acceptance_per_position_T0_pooled": {
        "dgx-01": {"control": [0.849, 0.699, 0.575, 0.472], "arm": [0.860, 0.720, 0.598, 0.498]},
        "dgx-02": {"control": [0.852, 0.703, 0.578, 0.474], "arm": [0.857, 0.717, 0.596, 0.493]}},
    "cells_arm_vs_control_pooled": {
        "dgx-01": {"fresh_c1": cell(1.51, 3.25), "fresh_c4": cell(3.44, 2.36), "fresh_c8": cell(3.26, 1.01),
                   "d16k_c4": cell(4.63, 2.51), "count_c8": cell(0.66, 1.01), "d16k_c8_wall": cell(0.41, 1.00),
                   "pp2048_c1": cell(0.46, 2.86), "tg512_c1": cell(0.66, 11.04), "tg512_c8": cell(1.22, 8.74)},
        "dgx-02": {"fresh_c1": cell(1.62, 8.29), "fresh_c4": cell(1.93, 4.77), "fresh_c8": cell(1.30, 2.76),
                   "d16k_c4": cell(1.98, 3.93), "count_c8": cell(1.04, 1.90), "d16k_c8_wall": cell(0.42, 1.00),
                   "pp2048_c1": cell(-0.67, 4.53), "tg512_c1": cell(2.95, 10.23), "tg512_c8": cell(2.29, 8.78)}},
    "benchy_boot_level_welch_ci95_pct": {"dgx-01": {"pp2048_c1": [-0.84, 1.76], "tg512_c1": [-5.25, 6.58], "tg512_c8": [-2.36, 4.80]},
                                         "dgx-02": {"pp2048_c1": [-2.44, 1.09], "tg512_c1": [-5.84, 11.74], "tg512_c8": [-2.07, 6.64]}},
    "history": "k56c alone: dgx-01 tg512 c8 -3.79% (noise 3.05%, 2 boots) made it INCONCLUSIVE; with 4 more boots per side the pooled cell is +1.22% (noise 8.74%). k56d dgx-02 fresh-c8 alone +2.12% (noise 1.42%).",
    "pooled_verdict": {"dgx-01": "PROMOTE (rule PASS)", "dgx-02": "INCONCLUSIVE: no cell beyond noise, none worse (rule FLAT)"},
}
losses = [f"{s}:{k}" for s, c in facts["cells_arm_vs_control_pooled"].items() for k, v in c.items() if v["beyond_noise"] and v["pct"] < 0]
if not facts["quality_gate"]["pass"]:
    verdict, conf, note = "reject", 1.0, "quality gate failed (forced, no Jev call)"
else:
    key = subprocess.run(["security", "find-generic-password", "-s", "dev/typesafe-ai-api-key", "-w"],
                         capture_output=True, text=True, check=True).stdout.strip()
    with TypeSafeClient(api_key=key) as client:
        r = client.system_one({"rules": RULES, "facts": facts}, {"verdict": Choice(
            instructions=("Given `rules` and `facts` (candidate `facts.arm` vs `facts.baseline` on two Sparks; "
                          "`facts.quality_gate.pass` is computed and non-negotiable; each cell carries pct, noise_pct "
                          "and beyond_noise), should this drafter replace the shipped one?"),
            criteria={"ship": "Gate passes, at least one beyond-noise win, no beyond-noise loss on either Spark.",
                      "ship_with_caveat": "Gate passes and there is a win, but there is a beyond-noise loss at c5+ or the picture is borderline.",
                      "reject": "Gate fails, or no beyond-noise win, or a non-trivial beyond-noise loss at c1-c4.",
                      "rerun": "The evidence is too thin or too noisy to judge."})})
    a = r.choices["verdict"]; verdict, conf, note = a.choice, a.confidence, None
    if losses and verdict == "ship":
        verdict, note = "ship_with_caveat", f"capped from ship: losses {losses}"
out = {**facts, "rules": RULES, "losses_beyond_noise": losses, "verdict": verdict, "confidence": conf,
       "cap_note": note, "judge": "Jev (TypeSafe System One), Choice", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(verdict, round(conf, 3), note)
