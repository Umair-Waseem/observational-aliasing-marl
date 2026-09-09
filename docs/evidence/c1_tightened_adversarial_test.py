"""Adversarial causal-soundness test for the TIGHTENED C1 metric (SHIFT 11, AM-1).

Reproduces the SHIFT-11 adversarial-audit findings and asserts the AM-1 own-reveal-
ablation witness closes them. Diagnostic-only (docs/evidence); writes no governed
artifact; reads no held-out layout (I-3).

Run:  python docs/evidence/c1_tightened_adversarial_test.py
Exits non-zero if any causal-soundness assertion fails.

The metric MUST:
  * PASS the scripted selective_sense exemplar (genuinely reveal-dependent safety).
  * REJECT F3 -- a privileged/memorized-SAFE route (reveal-INDEPENDENT) + incidental
    sensing + an incidental adapt-latch: the pooled composite graded it
    SELECTIVE_COMPOSITE, but its OWN reveal-ablation leaves hazard unchanged, so it is
    not a causal witness (verdict SENSES_NOT_CAUSAL).
  * REJECT F4 -- a near-always-sense (rate ~0.85) privileged-safe policy: not selective.
  * REJECT the s113-class failure-only sensing (covered by the real-checkpoint record).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Repository root: walk up from THIS FILE to the directory containing src/raas_marl.
# PHASE 3: both insertions used to come from a hardcoded absolute path with no walk
# and no existence guard, putting a private tree at sys.path[0] AND [1] -- so both
# raas_marl and the sibling harness loaded from there even when this script ran
# inside a clone. Both are now derived from __file__.
ROOT = Path(__file__).resolve()
while ROOT.parent != ROOT and not (ROOT / "src" / "raas_marl").is_dir():
    ROOT = ROOT.parent
if not (ROOT / "src" / "raas_marl").is_dir():
    raise RuntimeError(
        "could not locate the repository root: no ancestor directory of "
        f"{Path(__file__).resolve()} contains 'src/raas_marl'. Run this "
        "instrument from inside a checkout of the artifact repository."
    )
sys.path.insert(0, str(ROOT / "src"))
# The sibling instrument sits next to this file; take its directory from __file__
# directly rather than reconstructing it from ROOT.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c1_selectivity_harness as H  # noqa: E402
from raas_marl.environments.active_sensing.stage24_diagnostics import (  # noqa: E402
    _next_shortest_step,
    _movement_action,
    _position_from_observation,
    STAGE23_SENSE_ACTION_INDEX,
    SENSING_ACTION_FIELD,
    MOVEMENT_ACTION_FIELD,
    _FORK_WALL_COLUMN,
)


def _gates(scenario):
    obs = set(scenario.obstacles)
    return sorted(
        (r, _FORK_WALL_COLUMN)
        for r in range(scenario.height)
        if (r, _FORK_WALL_COLUMN) not in obs
    )


def privileged_safe(
    sensing_on: bool = True, near_always: bool = False
) -> Callable[[], Callable[..., dict[str, dict[str, int]]]]:
    """F3/F4: route is SAFE via PRIVILEGED hazard info (reveal-independent); sensing is
    incidental; the adapt-latch fires on a decoupled public-route counterfactual."""

    def _factory():
        def _p(observations, env, state, rng):
            scenario = env.scenario
            hazards = tuple(scenario.hidden_hazard_cells)  # privileged
            actions = {}
            for a in env.agents:
                pos = _position_from_observation(observations[a])
                revealed = tuple(sorted(state["revealed_hazards_by_agent"][a]))
                zone = set(state["public_risk_zone_cells"])
                sensed = state["sensed_positions_by_agent"][a]
                safe_next = _next_shortest_step(pos, scenario, avoid_hazards=hazards)
                if revealed:
                    pub = _next_shortest_step(pos, scenario, avoid_hazards=())
                    rev = _next_shortest_step(pos, scenario, avoid_hazards=revealed)
                    if pub != rev:
                        state["used_revealed_information_to_adapt_movement"] = True
                if near_always:
                    do_sense = sensing_on and pos[1] < 6
                else:
                    do_sense = (
                        sensing_on
                        and pos in zone
                        and pos not in sensed
                        and pos not in set(scenario.goals)
                    )
                    if do_sense:
                        sensed.add(pos)
                actions[a] = {
                    SENSING_ACTION_FIELD: STAGE23_SENSE_ACTION_INDEX if do_sense else 0,
                    MOVEMENT_ACTION_FIELD: _movement_action(pos, safe_next),
                }
            return actions

        return _p

    return _factory


def main() -> int:
    fam = "fork"

    sel = H.grade_policy(lambda: H._selective_sense_policy, label="selective_sense", family=fam)
    # privileged_safe(...) returns a fresh-policy FACTORY (what grade_policy expects).
    f3 = H.grade_policy(privileged_safe(sensing_on=True), label="F3_privileged_safe", family=fam)
    f3off = H.grade_policy(privileged_safe(sensing_on=False), label="F3_sensing_off", family=fam)
    f4 = H.grade_policy(privileged_safe(sensing_on=True, near_always=True), label="F4_near_always", family=fam)

    sel_tc = sel["tightened_composite"]
    f3_tc = f3["tightened_composite"]
    f4_tc = f4["tightened_composite"]
    sel_v = sel_tc["tightened_verdict"]
    f3_v = f3_tc["tightened_verdict"]
    f4_v = f4_tc["tightened_verdict"]
    haz_on = {k: v["hazard_entry_count"] for k, v in f3["per_surface"].items()}
    haz_off = {k: v["hazard_entry_count"] for k, v in f3off["per_surface"].items()}
    safety_identical = haz_on == haz_off

    checks = [
        ("selective_sense PASSES (genuine reveal-dependent)", sel_v == "SELECTIVE_COMPOSITE"),
        ("F3 privileged-safe REJECTED (own-ablation closes it)", f3_v != "SELECTIVE_COMPOSITE"),
        ("F3 sensing ON vs OFF hazard identical (sensing causally irrelevant)", safety_identical),
        ("F4 near-always REJECTED (selectivity ceiling / non-causal)", f4_v != "SELECTIVE_COMPOSITE"),
        # AM-3 (SHIFT-14): independent per-component reporting must reconcile with the verdict
        # for every exemplar (no report/verdict drift), and the positionally-induced annotation
        # must fire on F3 (the concrete densifier failure mode: succeeds + senses in-zone + NO
        # causal witness) while a genuine causal pass (selective_sense) is NEVER flagged.
        (
            "AM-3 reconcile: selective_sense/F3/F4 verdicts == per-component re-derivation",
            sel_tc["reconciliation"]["reconciles"]
            and f3_tc["reconciliation"]["reconciles"]
            and f4_tc["reconciliation"]["reconciles"],
        ),
        (
            "SHIFT-14: F3 flagged POSITIONALLY-INDUCED (in-zone sensing, zero causal witness)",
            f3_tc["positionally_induced"]["suspected"] is True
            and f3_tc["component_pass_counts"]["causal"] == 0,
        ),
        (
            "SHIFT-14: selective_sense NOT positionally-induced (causal witness is the firewall)",
            sel_tc["positionally_induced"]["suspected"] is False
            and sel_tc["component_pass_counts"]["causal"] >= 1,
        ),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    selective_sense={sel_v}  F3={f3_v}  F4={f4_v}  F3_safety_identical={safety_identical}")

    if all(ok for _, ok in checks):
        print("ADVERSARIAL CAUSAL-SOUNDNESS: ALL PASS")
        return 0
    print("ADVERSARIAL CAUSAL-SOUNDNESS: FAILURE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
