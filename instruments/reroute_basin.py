"""The reroute-basin classifier.

WHAT IT CLASSIFIES

On the lower validation mirror (`risk_fork_readiness_lower`) each trained policy
executes some reroute around the hazardous gate. This module labels which of a
small set of behavioural basins that reroute falls into. The paper reports the
resulting partition over the nine replication seeds in Table III:

    equivariant / overshoot / freeze  =  4/9 / 2/9 / 3/9

and, in Sec. V-D, states it as "the correct, equivariant rule appears on four of
nine seeds while the good outcome, which requires both mirrors, appears on only
two". The response letter (R1-S9) promises this classifier "validated to
reproduce the reported equivariant/overshoot/freeze partition"; running this
module performs exactly that validation and exits non-zero if it fails.

THE BASINS

    EQUIVARIANT  The intended rule. The policy crosses at the *perceived* open
                 safe gate -- row 1 on this mirror -- and takes no hazard. This
                 is the rule that would transfer, because it is keyed on what
                 the policy observes rather than on a memorised absolute row.
    OVERSHOOT    The memorised-absolute failure. The policy climbs past the
                 safe gate to the extreme grid-boundary row 0, which is where
                 the safe gate sat in the geometry it was trained on.
    FREEZE       The policy never crosses at all and runs out the episode
                 (>= 30 of the 32 available steps).
    BLIND        Crosses, but eats hazard: it did not use the reveal.
    OTHER        None of the above; no seed in the reported set falls here.

WHICH FIELDS IT READS

Five fields of the policy's rollout on the lower validation mirror:

    any_row0        bool  -- did either agent ever occupy grid row 0?
    cross_rows      {agent -> row or null} -- the wall-column row each agent
                    crossed at, or null if that agent never crossed
    hazard_entries  int   -- hazard cells entered on the episode
    steps           int   -- episode length
    team_success    bool  -- read, but see the fidelity note below

In this repository those five live, per seed, at

    docs/evidence/t1_seedscale_analysis.json
        -> per_seed[<seed>]["readiness_lower_outcome"]

where `team_success` and `hazard_entries` appear under their shorter recorded
names `succ` and `haz`. `SHIPPED_FIELD_ALIASES` below is that mapping, applied
by `classify_record`, so the classifier runs against either spelling.

FIDELITY NOTE

`classify` reproduces the original decision procedure branch for branch and in
the original order -- early returns included, since the order is what decides a
policy satisfying more than one predicate. The original bound `team_success` and
then never tested it; that is preserved deliberately rather than "fixed",
because changing it could move a seed, and the published partition is what this
module exists to reproduce. The parameter is accepted and documented as unused.

USAGE

    python -m instruments.reroute_basin            # classify and validate
    python -m instruments.reroute_basin --json     # machine-readable output

Standard library only; imports nothing from `raas_marl`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "BASINS",
    "REPORTED_PARTITION",
    "SHIPPED_FIELD_ALIASES",
    "classify",
    "classify_record",
    "classify_shipped_seeds",
    "load_shipped_outcomes",
    "validate",
]

#: The label set, in the order Table III's row lists the first three.
BASINS = ("EQUIVARIANT", "OVERSHOOT", "FREEZE", "BLIND", "OTHER")

#: Where the shipped per-seed outcomes live, relative to the repository root.
ANALYSIS_JSON = Path("docs/evidence/t1_seedscale_analysis.json")

#: The nine replication seeds, in the order the paper names them (Sec. IV:
#: "seeds 147 to 149 and 153 to 158"; 150-152 were allocated to an experiment
#: cancelled before launch and never run).
SEEDS = ("147", "148", "149", "153", "154", "155", "156", "157", "158")

#: Recorded field name -> the name `classify` takes.
SHIPPED_FIELD_ALIASES = {"succ": "team_success", "haz": "hazard_entries"}

#: The partition printed in Table III. `validate` checks against this.
REPORTED_PARTITION: dict[str, tuple[str, ...]] = {
    "EQUIVARIANT": ("147", "153", "157", "158"),
    "OVERSHOOT": ("149", "154"),
    "FREEZE": ("148", "155", "156"),
}

#: Steps at or beyond which a non-crossing episode counts as a freeze rather
#: than a slow success. The episode horizon is 32.
FREEZE_STEP_FLOOR = 30

#: The wall-column row that is the *perceived* open safe gate on the lower
#: validation mirror.
SAFE_GATE_ROW = 1

_REQUIRED_FIELDS = ("any_row0", "cross_rows", "hazard_entries", "steps")


def classify(
    *,
    any_row0: bool,
    cross_rows: Mapping[str, Any],
    hazard_entries: int,
    steps: int,
    team_success: bool | None = None,
) -> str:
    """Return the reroute basin for one lower-validation-mirror rollout.

    `team_success` is accepted for signature fidelity with the original and is
    deliberately not consulted; see the module docstring's fidelity note.
    """
    del team_success  # bound by the original, never tested by it.

    crossed = any(v is not None for v in cross_rows.values())
    crossed_at_safe_gate = any(v == SAFE_GATE_ROW for v in cross_rows.values())

    if any_row0:
        return "OVERSHOOT"
    if crossed_at_safe_gate and hazard_entries == 0:
        return "EQUIVARIANT"
    if not crossed and steps >= FREEZE_STEP_FLOOR:
        return "FREEZE"
    if hazard_entries > 0:
        return "BLIND"
    return "OTHER"


def classify_record(record: Mapping[str, Any]) -> str:
    """Classify one recorded outcome, accepting either field spelling."""
    fields = {SHIPPED_FIELD_ALIASES.get(k, k): v for k, v in record.items()}
    missing = [name for name in _REQUIRED_FIELDS if name not in fields]
    if missing:
        raise KeyError(f"outcome record is missing {missing}")
    return classify(
        any_row0=bool(fields["any_row0"]),
        cross_rows=fields["cross_rows"],
        hazard_entries=int(fields["hazard_entries"]),
        steps=int(fields["steps"]),
        team_success=fields.get("team_success"),
    )


def _repo_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    return Path(__file__).resolve().parents[1]


def load_shipped_outcomes(root: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Read `readiness_lower_outcome` for the nine seeds from the shipped JSON."""
    payload = json.loads(
        (_repo_root(root) / ANALYSIS_JSON).read_text(encoding="utf-8")
    )
    per_seed = payload["per_seed"]
    return {seed: per_seed[seed]["readiness_lower_outcome"] for seed in SEEDS}


def classify_shipped_seeds(root: Path | str | None = None) -> dict[str, str]:
    """Classify all nine seeds from the shipped records."""
    return {
        seed: classify_record(outcome)
        for seed, outcome in load_shipped_outcomes(root).items()
    }


def _partition(labels: Mapping[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for seed, label in labels.items():
        out.setdefault(label, []).append(seed)
    return {key: sorted(value, key=int) for key, value in sorted(out.items())}


def validate(root: Path | str | None = None) -> tuple[bool, dict[str, list[str]]]:
    """Classify the nine seeds and compare with the partition Table III prints."""
    got = _partition(classify_shipped_seeds(root))
    expected = {k: sorted(v, key=int) for k, v in REPORTED_PARTITION.items()}
    return got == expected, got


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify the reroute basin of each replication seed."
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--root", default=None, help="repository root (default: inferred)")
    args = parser.parse_args(argv)

    labels = classify_shipped_seeds(args.root)
    ok, got = validate(args.root)

    if args.json:
        print(json.dumps(
            {"per_seed": labels, "partition": got, "matches_paper": ok}, indent=1
        ))
        return 0 if ok else 1

    outcomes = load_shipped_outcomes(args.root)
    print("Reroute basin on the lower validation mirror (risk_fork_readiness_lower)")
    print()
    header = "  {:>5}  {:>6}  {:<20}  {:>4}  {:>5}  {}".format(
        "seed", "row0", "cross_rows", "haz", "steps", "basin"
    )
    print(header)
    for seed in SEEDS:
        outcome = outcomes[seed]
        crossings = ", ".join(
            "{}:{}".format(agent.split("_")[-1], row)
            for agent, row in sorted(outcome["cross_rows"].items())
        )
        hazard = outcome.get("haz", outcome.get("hazard_entries"))
        print("  {:>5}  {:>6}  {:<20}  {:>4}  {:>5}  {}".format(
            seed, str(bool(outcome["any_row0"])), crossings,
            int(hazard), int(outcome["steps"]), labels[seed],
        ))
    print()
    for label in BASINS:
        seeds = got.get(label, [])
        if seeds or label in REPORTED_PARTITION:
            print("  {:<12} {}/9   {{{}}}".format(label, len(seeds), ", ".join(seeds)))
    print()
    print("  Table III prints  equivariant / overshoot / freeze = 4/9 / 2/9 / 3/9")
    print("  " + ("MATCHES the reported partition."
                  if ok else "*** DOES NOT MATCH THE REPORTED PARTITION ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
