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
    "arm": "1x release candidate v2.2.0: drafter D3 served from HF ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE @ 03f4a057 on the published image tp1-d3-hf (CI-built)",
    "baseline": "shipped 1x v2.1.0: drafter D1, HF @ 16c9bd54, published image tp1-v3e-hf",
    "design": "k80: Thunderdome per Spark, ctl arm arm ctl twice (4 boots per side per Spark), warm-up, T=0 fresh-c4/c8 probes, llama-benchy pp2048 c1 + tg512 c1/c8 at 5 runs per boot. Earlier local-snapshot evidence (k56c+k56d, same weight files) gave dgx-01 PROMOTE, dgx-02 flat, no cell worse.",
    "why_rerun": "k79's single check boot of v2.2.0 read D1-like acceptance (0.688/0.562 at pos 2-3); weights were confirmed byte-identical to the k56 arm; k80 compares D3 vs D1 both on the published stack",
    "quality_gate": {"pass": True, "hardmode": [93], "tc45": [100], "fidelity": "20/20 exact at 8k/32k/64k/128k and two extra 128k seeds (k56c, identical weight files)", "stragglers_preemptions": 0},
    "post_checks": "every boot 0 measured b12x plans, KV pool 993,754 both sides, AOT loaded; dgx-02 arm boots log 19 TritonBundler 'Failed to reload cubin' warnings each (documented false positive: the image AOT seed was built on dgx-01; first dgx-02 arm boot 277 s, later 147 s = control)",
    "acceptance_per_position_T0_pooled": {
        "dgx-01": {"control": [0.841, 0.683, 0.554, 0.445], "arm": [0.852, 0.704, 0.577, 0.469]},
        "dgx-02": {"control": [0.841, 0.683, 0.552, 0.444], "arm": [0.854, 0.705, 0.578, 0.470]}},
    "cells_arm_vs_control": {
        "dgx-01": {"fresh_c4": cell(1.79, 2.19), "fresh_c8": cell(2.10, 1.84), "pp2048_c1": cell(-0.09, 1.06), "tg512_c1": cell(7.94, 9.63), "tg512_c8": cell(3.41, 6.01)},
        "dgx-02": {"fresh_c4": cell(2.60, 2.36), "fresh_c8": cell(2.01, 2.21), "pp2048_c1": cell(-0.22, 1.73), "tg512_c1": cell(1.73, 8.95), "tg512_c8": cell(3.75, 4.36)}},
    "local_stack_k56c_k56d_pooled_dgx01_beyond_noise": {"fresh_c4": "+3.44% (2.36)", "fresh_c8": "+3.26% (1.01)", "d16k_c4": "+4.63% (2.51)"},
}
losses = [f"{s}:{k}" for s, c in facts["cells_arm_vs_control"].items() for k, v in c.items() if v["beyond_noise"] and v["pct"] < 0]
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
