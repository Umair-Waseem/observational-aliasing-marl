"""The grade composer: per-checkpoint grade records -> the paper's Table III rates.

The response letter (R1-S9) states that "the reported reliability rates recompute
from the shipped per-checkpoint grade records via the released composer". This
module is that composer. It reads only records shipped in this repository and
recomputes the four n=9 reliability rates the paper prints in Table III.

THE TWO-OF-THREE RULE

Each of the nine replication seeds was graded at three checkpoints -- rounds
3500, 3750 and 4000 -- and a seed carries a witness-based conjunct only if it
holds at at least TWO of those three. That majority-over-checkpoints rule is
what makes a rate a property of a converged policy rather than of the single
round it happened to be graded at. Four conjuncts are composed this way:

    retain_c   verdict is SELECTIVE_COMPOSITE with a witness on >= 1 TRAINED
               surface (the {3,4} anchor pair or the {0,3} curriculum pair)
    anchor_c   verdict is SELECTIVE_COMPOSITE with a witness on >= 1 ANCHOR
               surface (risk_fork_train_lower / _upper)
    c4_gen     BOTH validation mirrors succeed with zero hazard entries at the
               SAME checkpoint
    c1_gen     a witness on >= 1 validation mirror  (reported, never a gate)

THE FOUR RATES, AND WHERE EACH CONJUNCT COMES FROM

This is the "what recomputes from what" the letter promises, stated exactly.
Two of the four criteria are conjunctions that draw on both shipped record
kinds; the composer takes each conjunct from its own source and combines them.

    Task success        5/9   tail-100 team success >= 0.9 AND episodic hazard
                              <= 0.5.  Source: results/held2_conjuncts.json
                              (a training-curve quantity; no checkpoint record
                              enters this one).
    Core hold           4/9   HELD-2 (a) joint occupancy AND (b) two-sided
                              relapse windows  [results/held2_conjuncts.json]
                              AND the task-success bar  [same file]
                              AND retain_c at 2 of 3   [checkpoint records].
    Anchor consolidation 7/9  both anchor mirrors' tail-100 hazard <= 0.5
                              [t1_seedscale_analysis.json :: per_mirror]
                              AND anchor_c at 2 of 3   [checkpoint records].
    Transfer            2/9   c4_gen at 2 of 3         [checkpoint records only].

So the transfer rate -- the paper's headline negative -- recomputes from the
per-checkpoint records alone. The other three each add one curve-derived
conjunct, which is shipped alongside them.

INPUTS (all inside this repository)

    results/experiments/<run_id>/checkpoint_grades/grade_r{3500,3750,4000}.json
        27 records: the 9 seeds x 3 checkpoints. Each holds the full grade the
        certification harness produced, from which this module re-derives the
        per-checkpoint conjuncts rather than trusting a stored boolean.
    results/held2_conjuncts.json
        the curve-derived conjuncts named above.
    docs/evidence/t1_seedscale_analysis.json
        per-mirror tail-100 hazard, and the registered per-seed values that
        `--check` audits this recomputation against.

FRESH-SEED-ONLY RATES

Table III also prints each rate over the six seeds that were fresh to the
replication (153-158), the anchors 147-149 having been graded once before.
Those are the same computation restricted to that subset.

USAGE

    python -m instruments.grade_composer            # the table
    python -m instruments.grade_composer --check    # also audit vs the registry
    python -m instruments.grade_composer --json

Standard library only; imports nothing from `raas_marl`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ANCHOR_SURFACES",
    "CHECKPOINT_ROUNDS",
    "REPORTED_RATES",
    "CRITERION_LABELS",
    "TRAINED_SURFACES",
    "VALIDATION_SURFACES",
    "compose_seed",
    "compose_all",
    "derive_checkpoint",
    "tallies",
]

#: The three graded checkpoints, in round order.
CHECKPOINT_ROUNDS = (3500, 3750, 4000)

#: The nine replication seeds and the run that produced each.
SEED_RUNS: dict[str, str] = {
    "147": "t1_retain_s147", "148": "t1_retain_s148", "149": "t1_retain_s149",
    "153": "t1_seedscale_s153", "154": "t1_seedscale_s154", "155": "t1_seedscale_s155",
    "156": "t1_seedscale_s156", "157": "t1_seedscale_s157", "158": "t1_seedscale_s158",
}

#: The six seeds fresh to the replication (the anchors 147-149 were graded before).
FRESH_SEEDS = ("153", "154", "155", "156", "157", "158")

#: The two mirrors of the {3,4} anchor pair.
ANCHOR_SURFACES = frozenset({"risk_fork_train_lower", "risk_fork_train_upper"})

#: The {0,3} curriculum pair, the second trained pair in the mixture.
CURRICULUM_R0R3_SURFACES = frozenset({
    "risk_fork_curriculum_r0r3_lower", "risk_fork_curriculum_r0r3_upper",
})

#: Every surface the mixture actually trained on. The certification harness also
#: enumerates the r4r7 and r0r7 curriculum pairs, which this run set did NOT
#: train; those are informative-only and are never a witness for `retain_c`.
TRAINED_SURFACES = ANCHOR_SURFACES | CURRICULUM_R0R3_SURFACES

#: The two validation mirrors (gate rows 1 and 2). The held-out geometry (gate
#: rows 5 and 6) is not among them: it was never instantiated, so no record in
#: this repository refers to it.
VALIDATION_SURFACES = ("risk_fork_readiness_lower", "risk_fork_readiness_upper")

#: The verdict the certification harness emits for a policy that senses
#: selectively AND causally (own-ablation confirmed).
SELECTIVE_VERDICT = "SELECTIVE_COMPOSITE"

#: A seed carries a witness conjunct if it holds at >= this many checkpoints.
MAJORITY = 2

#: Table III's n=9 column, as printed. `--check` compares against these.
REPORTED_RATES: dict[str, tuple[int, int]] = {
    "task_success": (5, 9),
    "core_hold": (4, 9),
    "anchor_consolidation": (7, 9),
    "transfer": (2, 9),
}

#: Table III's fresh-seed-only rows, as printed.
REPORTED_FRESH_RATES: dict[str, tuple[int, int]] = {
    "task_success": (3, 6),
    "core_hold": (2, 6),
    "anchor_consolidation": (4, 6),
    "transfer": (1, 6),
}

#: Bars for the anchor-consolidation hazard conjunct.
ANCHOR_HAZARD_CEILING = 0.5

#: `t1_seedscale_analysis.json`'s `per_mirror` block keys the two anchor mirrors
#: by their short names, without the `risk_fork_` prefix the grade records use.
ANCHOR_PER_MIRROR_KEYS = ("train_lower", "train_upper")

CRITERIA = ("task_success", "core_hold", "anchor_consolidation", "transfer")

CRITERION_LABELS = {
    "task_success": "Task success (joint with hazard)",
    "core_hold": "Selective-sensing hold (core)",
    "anchor_consolidation": "Anchor consolidation (load-bearing)",
    "transfer": "Transfer to validation",
}


def _repo_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    return Path(__file__).resolve().parents[1]


def derive_checkpoint(record: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive one checkpoint's conjuncts from its shipped grade record.

    The record stores the harness's full grade. The booleans below are computed
    here rather than read, so that the composition is auditable end to end from
    what the harness measured (per-surface success, hazard and witness lists)
    rather than from a summary somebody else already reduced.
    """
    grade = record["grade"]
    composite = grade["tightened_composite"]
    per_surface = grade["per_surface"]

    witnesses = [entry["surface"] for entry in (composite.get("witness_surfaces") or [])]
    selective = composite["tightened_verdict"] == SELECTIVE_VERDICT

    validation_pass = {
        surface: bool(per_surface[surface]["team_success"])
        and int(per_surface[surface]["hazard_entry_count"]) == 0
        for surface in VALIDATION_SURFACES
    }

    return {
        "verdict": composite["tightened_verdict"],
        "witness_surfaces": witnesses,
        "retain_c": selective and any(s in TRAINED_SURFACES for s in witnesses),
        "anchor_c": selective and any(s in ANCHOR_SURFACES for s in witnesses),
        "c4_gen": all(validation_pass.values()),
        "c4_gen_by_surface": validation_pass,
        "c1_gen": any(s in VALIDATION_SURFACES for s in witnesses),
    }


def load_checkpoints(seed: str, root: Path | str | None = None) -> dict[int, dict[str, Any]]:
    """Load and derive all three checkpoint records for one seed."""
    run = SEED_RUNS[seed]
    base = _repo_root(root) / "results" / "experiments" / run / "checkpoint_grades"
    out: dict[int, dict[str, Any]] = {}
    for rnd in CHECKPOINT_ROUNDS:
        path = base / f"grade_r{rnd:06d}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        out[rnd] = derive_checkpoint(record)
    return out


def compose_seed(seed: str, root: Path | str | None = None) -> dict[str, Any]:
    """Compose one seed's four criteria from the shipped records."""
    root_path = _repo_root(root)
    checkpoints = load_checkpoints(seed, root_path)

    def majority(key: str) -> tuple[int, bool]:
        count = sum(1 for c in checkpoints.values() if c[key])
        return count, count >= MAJORITY

    retain_n, retain_pass = majority("retain_c")
    anchor_n, anchor_pass = majority("anchor_c")
    c4gen_n, c4gen_pass = majority("c4_gen")
    c1gen_n, c1gen_pass = majority("c1_gen")

    conjuncts = json.loads(
        (root_path / "results" / "held2_conjuncts.json").read_text(encoding="utf-8")
    )["per_seed"][seed]

    analysis = json.loads(
        (root_path / "docs" / "evidence" / "t1_seedscale_analysis.json").read_text(
            encoding="utf-8"
        )
    )["per_seed"][seed]
    per_mirror = analysis["per_mirror"]
    anchor_hazard_ok = all(
        float(per_mirror[mirror]["haz"]) <= ANCHOR_HAZARD_CEILING
        for mirror in ANCHOR_PER_MIRROR_KEYS
    )

    task_success = bool(conjuncts["bar_c4_pass"])
    core_hold = bool(conjuncts["held2_ab_pass"]) and task_success and retain_pass
    anchor_consolidation = anchor_hazard_ok and anchor_pass
    transfer = c4gen_pass

    return {
        "seed": seed,
        "run": SEED_RUNS[seed],
        "checkpoints": {str(k): v for k, v in checkpoints.items()},
        "counts_of_three": {
            "retain_c": retain_n, "anchor_c": anchor_n,
            "c4_gen": c4gen_n, "c1_gen": c1gen_n,
        },
        "two_of_three": {
            "retain_c": retain_pass, "anchor_c": anchor_pass,
            "c4_gen": c4gen_pass, "c1_gen": c1gen_pass,
        },
        "curve_conjuncts": {
            "held2_ab_pass": bool(conjuncts["held2_ab_pass"]),
            "bar_c4_pass": task_success,
            "bar_c4_tail100_success": conjuncts["bar_c4_tail100_success"],
            "bar_c4_tail100_hazard": conjuncts["bar_c4_tail100_hazard"],
            "anchor_mirrors_hazard_le_0_5": anchor_hazard_ok,
        },
        "criteria": {
            "task_success": task_success,
            "core_hold": core_hold,
            "anchor_consolidation": anchor_consolidation,
            "transfer": transfer,
        },
    }


def compose_all(root: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Compose every replication seed."""
    return {seed: compose_seed(seed, root) for seed in SEED_RUNS}


def tallies(
    composed: Mapping[str, Mapping[str, Any]], seeds: tuple[str, ...] | None = None
) -> dict[str, tuple[int, int]]:
    """Count how many of `seeds` pass each criterion."""
    chosen = tuple(seeds) if seeds is not None else tuple(composed)
    return {
        criterion: (
            sum(1 for s in chosen if composed[s]["criteria"][criterion]), len(chosen)
        )
        for criterion in CRITERIA
    }


def audit(composed: Mapping[str, Mapping[str, Any]], root: Path | str | None = None) -> list[str]:
    """Compare this recomputation with the values registered in the analysis JSON.

    Returns a list of disagreements; empty means the composer reproduces the
    registry seed by seed, not merely in aggregate.
    """
    registered = json.loads(
        (_repo_root(root) / "docs" / "evidence" / "t1_seedscale_analysis.json").read_text(
            encoding="utf-8"
        )
    )["per_seed"]
    problems: list[str] = []
    mapping = {
        "task_success": lambda r: bool(r["bar_c4_retain"]["pass"]),
        "core_hold": lambda r: bool(r["held2_full"]),
        "anchor_consolidation": lambda r: bool(r["anchor_clean_class"]),
        "transfer": lambda r: bool(r["c4_gen_2of3"]),
    }
    for seed, entry in composed.items():
        record = registered[seed]
        for criterion, getter in mapping.items():
            want, got = getter(record), entry["criteria"][criterion]
            if want != got:
                problems.append(
                    f"seed {seed}: {criterion} recomputed {got}, registry has {want}"
                )
    return problems


def _fmt(count: int, total: int) -> str:
    return f"{count}/{total} = {count / total:.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute Table III's reliability rates from the shipped records."
    )
    parser.add_argument("--check", action="store_true",
                        help="also audit against the registered per-seed values")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--root", default=None, help="repository root (default: inferred)")
    args = parser.parse_args(argv)

    composed = compose_all(args.root)
    n9 = tallies(composed)
    n6 = tallies(composed, FRESH_SEEDS)
    problems = audit(composed, args.root) if args.check else []
    matches = all(n9[c] == REPORTED_RATES[c] for c in CRITERIA) and all(
        n6[c] == REPORTED_FRESH_RATES[c] for c in CRITERIA
    )

    if args.json:
        print(json.dumps({
            "per_seed": {s: v["criteria"] for s, v in composed.items()},
            "counts_of_three": {s: v["counts_of_three"] for s, v in composed.items()},
            "n9": {k: list(v) for k, v in n9.items()},
            "fresh_seed_only": {k: list(v) for k, v in n6.items()},
            "matches_paper": matches,
            "registry_disagreements": problems,
        }, indent=1))
        return 0 if matches and not problems else 1

    print("Per-seed composition (counts are of the three checkpoints r3500/r3750/r4000)")
    print()
    print("  {:>5}  {:>19}  {:>8} {:>8} {:>7} {:>7}   {}".format(
        "seed", "run", "retain_c", "anchor_c", "c4_gen", "c1_gen", "criteria met"))
    for seed, entry in composed.items():
        counts = entry["counts_of_three"]
        met = [c for c in CRITERIA if entry["criteria"][c]]
        print("  {:>5}  {:>19}  {:>8} {:>8} {:>7} {:>7}   {}".format(
            seed, entry["run"],
            f"{counts['retain_c']}/3", f"{counts['anchor_c']}/3",
            f"{counts['c4_gen']}/3", f"{counts['c1_gen']}/3",
            ", ".join(met) if met else "-"))
    print()
    print("  {:<36} {:>14} {:>14} {:>14}".format(
        "Criterion", "n=9", "fresh-only", "Table III n=9"))
    for criterion in CRITERIA:
        got, reported = n9[criterion], REPORTED_RATES[criterion]
        flag = "" if got == reported else "   <-- MISMATCH"
        print("  {:<36} {:>14} {:>14} {:>14}{}".format(
            CRITERION_LABELS[criterion], _fmt(*got), _fmt(*n6[criterion]),
            f"{reported[0]}/{reported[1]}", flag))
    print()
    if args.check:
        if problems:
            print("  REGISTRY AUDIT: {} disagreement(s)".format(len(problems)))
            for line in problems:
                print("    -", line)
        else:
            print("  REGISTRY AUDIT: every seed agrees with the registered per-seed values.")
    print("  " + ("MATCHES Table III (both the n=9 and the fresh-seed-only columns)."
                  if matches else "*** DOES NOT MATCH TABLE III ***"))
    return 0 if matches and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
