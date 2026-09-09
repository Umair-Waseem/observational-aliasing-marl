"""Reproduce Table III of the paper, end to end, from the shipped records.

One command that runs all three instruments and prints, beside each number, the
value the camera-ready prints:

  * the four n=9 reliability rates and the four fresh-seed-only rates
        -> instruments.grade_composer, over the 27 per-checkpoint grade records
           and the curve-derived conjuncts
  * their exact 95% Clopper-Pearson intervals
        -> instruments.statistics
  * the reroute-basin partition
        -> instruments.reroute_basin, over the shipped per-seed outcomes
  * the arm-fire de-confound Fisher p-value
        -> instruments.statistics, over counts this module derives from the
           shipped analysis JSON rather than taking on trust

Exits 0 only if every reproduced value matches the paper. Any mismatch is
printed and the exit status is 1, so this is usable as a regression gate.

    python -m instruments.reproduce_table3

Standard library only: no torch, no scipy, no network. Nothing here imports
`raas_marl`, so the numbers can be checked without installing the package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import grade_composer as composer
from . import reroute_basin as basin
from . import statistics as stats

#: The intervals Table III prints, to the three decimals it prints them at.
REPORTED_INTERVALS = {
    "task_success": (0.212, 0.863),
    "core_hold": (0.137, 0.788),
    "anchor_consolidation": (0.400, 0.972),
    "transfer": (0.028, 0.600),
}

#: The de-confound p-value printed in the note to Table III.
REPORTED_FISHER_P = 0.417

_RULE = "-" * 78


def fisher_arm_fire_deconfound(root: Path | str | None = None) -> tuple[tuple[int, int, int, int], float, dict]:
    """Build the arm-fire x transfer 2x2 from the shipped records and test it.

    The paper's note to Table III reports "no feature (the arm-fire signature)
    that separates the two transferring seeds (Fisher p=0.417)". The 2x2 is
    built here from the per-seed records rather than hard-coded: the arm-fire
    signature is `mech_f84_fired` and transfer is the composed `c4_gen` conjunct.
    """
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    per_seed = json.loads(
        (root_path / "docs" / "evidence" / "t1_seedscale_analysis.json").read_text(
            encoding="utf-8"
        )
    )["per_seed"]
    composed = composer.compose_all(root_path)

    fired = {s for s in composer.SEED_RUNS if per_seed[s]["mech_f84_fired"]}
    moved = {s for s in composer.SEED_RUNS if composed[s]["criteria"]["transfer"]}
    every = set(composer.SEED_RUNS)

    a = len(fired & moved)
    b = len(fired - moved)
    c = len(moved - fired)
    d = len(every - fired - moved)
    return (a, b, c, d), stats.fisher_exact_two_sided(a, b, c, d), {
        "arm_fire_seeds": sorted(fired, key=int),
        "transferring_seeds": sorted(moved, key=int),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce Table III of the paper from the shipped records."
    )
    parser.add_argument("--root", default=None, help="repository root (default: inferred)")
    args = parser.parse_args(argv)
    root = args.root
    failures: list[str] = []

    print(_RULE)
    print("TABLE III, REPRODUCED FROM THE SHIPPED RECORDS")
    print(_RULE)

    # ---- rates + intervals -------------------------------------------------
    composed = composer.compose_all(root)
    n9 = composer.tallies(composed)
    n6 = composer.tallies(composed, composer.FRESH_SEEDS)

    print()
    print("  {:<36} {:>12} {:>20} {:>10}".format(
        "Criterion", "n=9", "exact 95% interval", "fresh-only"))
    for criterion in composer.CRITERIA:
        k, n = n9[criterion]
        low, high = stats.clopper_pearson(k, n)
        fk, fn = n6[criterion]

        want_rate = composer.REPORTED_RATES[criterion]
        want_fresh = composer.REPORTED_FRESH_RATES[criterion]
        want_low, want_high = REPORTED_INTERVALS[criterion]

        if (k, n) != want_rate:
            failures.append(f"{criterion}: rate {k}/{n}, paper prints {want_rate[0]}/{want_rate[1]}")
        if (fk, fn) != want_fresh:
            failures.append(
                f"{criterion}: fresh-only {fk}/{fn}, paper prints {want_fresh[0]}/{want_fresh[1]}")
        if abs(round(low, 3) - want_low) > 5e-4 or abs(round(high, 3) - want_high) > 5e-4:
            failures.append(
                f"{criterion}: interval [{low:.3f}, {high:.3f}], "
                f"paper prints [{want_low:.3f}, {want_high:.3f}]")

        print("  {:<36} {:>12} {:>20} {:>10}".format(
            composer.CRITERION_LABELS[criterion],
            f"{k}/{n} = {k / n:.2f}",
            f"[{low:.3f}, {high:.3f}]",
            f"{fk}/{fn} = {fk / fn:.2f}",
        ))

    # ---- the transfer interval's containment of chance ---------------------
    tk, tn = n9["transfer"]
    t_low, t_high = stats.clopper_pearson(tk, tn)
    chance = 1.0 / 3.0
    contains = t_low <= chance <= t_high
    print()
    print(f"  The transfer interval [{t_low:.3f}, {t_high:.3f}] "
          f"{'CONTAINS' if contains else 'does NOT contain'} the chance level 1/3 = {chance:.3f}.")
    if not contains:
        failures.append("the transfer interval does not contain 1/3, which the paper states it does")

    # ---- reroute basins ----------------------------------------------------
    ok_basin, partition = basin.validate(root)
    print()
    print("  Reroute-basin classification (n=9):")
    print("    equivariant / overshoot / freeze  =  {}/9 / {}/9 / {}/9   (paper: 4/9 / 2/9 / 3/9)".format(
        len(partition.get("EQUIVARIANT", [])),
        len(partition.get("OVERSHOOT", [])),
        len(partition.get("FREEZE", [])),
    ))
    for label in ("EQUIVARIANT", "OVERSHOOT", "FREEZE"):
        print(f"      {label:<12} {{{', '.join(partition.get(label, []))}}}")
    if not ok_basin:
        failures.append("the reroute-basin partition does not match the reported one")

    # ---- Fisher de-confound ------------------------------------------------
    table, p_value, detail = fisher_arm_fire_deconfound(root)
    a, b, c, d = table
    print()
    print("  Arm-fire de-confound (note to Table III):")
    print(f"    arm-fire seeds     {detail['arm_fire_seeds']}")
    print(f"    transferring seeds {detail['transferring_seeds']}")
    print(f"    2x2 = [[{a}, {b}], [{c}, {d}]]   Fisher exact (two-sided) p = {p_value:.4f}"
          f"   (paper: {REPORTED_FISHER_P})")
    if abs(round(p_value, 3) - REPORTED_FISHER_P) > 5e-4:
        failures.append(f"Fisher p = {p_value:.4f}, paper prints {REPORTED_FISHER_P}")

    # ---- verdict -----------------------------------------------------------
    print()
    print(_RULE)
    if failures:
        print(f"FAILED: {len(failures)} value(s) do not match the paper.")
        for line in failures:
            print("  -", line)
        print(_RULE)
        return 1
    print("All reproduced values match the camera-ready.")
    print(_RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
