"""SHIFT 32 — rejection-existence checks for the NEW eval-relevant degenerate nulls
discovered T1-D12 -> T1-D15 (F-81..F-89), extending the R2 witness-bar inventory
(ED-Stage26_c1_metric_revision §11.2, built SHIFT 22 / T1-D11).

DRAFT groundwork: DIAGNOSTIC ONLY. Pins nothing, reads no held-out layout, grades
no run. Confirms the tightened surface-resolved causal composite verdict CASCADE
REJECTS each new null SHAPE (rejection-existence per R2's own standard: "a single
extremal instance certifies rejection-existence only"). Every null shape below is a
synthetic per-surface grade constructed to match a MEASURED ledger basin.

Run: python docs/evidence/r2_null_rejection_check.py            (prints a table)
     python docs/evidence/r2_null_rejection_check.py --json     (also writes the record)
     python docs/evidence/r2_null_rejection_check.py --json --out PATH   (record -> PATH)

--json writes to docs/evidence/r2_null_inventory_shift32.json inside the checkout this
file was run from; --out redirects it. Without --json nothing is written.
Exits non-zero on any surprise.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Repository root: walk up from THIS FILE to the directory containing src/raas_marl.
# PHASE 3: both insertions used to come from a hardcoded absolute path with no walk
# and no existence guard, putting a private tree at sys.path[0] AND [1] -- so both
# raas_marl and the sibling harness loaded from there even when this script ran
# inside a clone. Both are now derived from __file__. The --json output path is
# likewise relative to the resolved root (see _output_path) and overridable with
# --out: a public instrument must not write into anyone else's tree by default.
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

CEIL = H.SELECTIVITY_CEILING          # 0.5
PI_FLOOR = H.POSITIONALLY_INDUCED_INZONE_FLOOR  # 0.10


def _surface(*, success, total_sense, rate, adapt, own_avoided,
             hazard_avoided_vs_no_sense=0, inzone=None, group="train"):
    return {
        "team_success": success,
        "adapt_fired": adapt,
        "own_blind_hazard_avoided": own_avoided,
        "hazard_avoided_vs_no_sense": hazard_avoided_vs_no_sense,
        "group": group,
        "selectivity": {
            "total_sense": total_sense,
            "overall_sense_rate": rate,
            "contrast_strict_in_zone": {"p_sense_relevant": inzone},
        },
    }


def _grade(per_surface, *, n, total_sense):
    rate = total_sense / n if n else 0.0
    return {
        "per_surface": per_surface,
        "pooled_selectivity": {
            "n_decisions": n,
            "total_sense": total_sense,
            "overall_sense_rate": rate,
        },
        "composite": {"composite_verdict": "SYNTHETIC"},
    }


def verdict(grade: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Run one synthetic grade through the tightened C1 composite.

    Parameters
    ----------
    grade : dict[str, Any]
        A synthetic per-surface grade, as built by :func:`_grade`.

    Returns
    -------
    tuple[str, dict[str, Any]]
        The verdict string and the full composite payload behind it. The payload is
        heterogeneous by design -- its keys vary with which branch of the cascade
        fired -- so it is honestly typed ``dict[str, Any]`` rather than given a
        fabricated schema.
    """
    tc = H.tightened_composite_c1_verdict(grade)
    return tc["tightened_verdict"], tc


CHECKS = []


def check(name: str, got: str, want_not: set[str], extra: str = "") -> None:
    """Record a rejection-existence check.

    Parameters
    ----------
    name : str
        Human-readable description of what is being asserted.
    got : str
        The verdict the composite actually returned.
    want_not : set[str]
        Verdicts that would mean the null was NOT rejected; the check passes when
        ``got`` is outside this set.
    extra : str, optional
        Extra detail appended to the printed row.

    Returns
    -------
    None
        Appends to the module-level ``CHECKS`` list; :func:`main` prints and grades it.
    """
    ok = got not in want_not
    CHECKS.append((name, ok, f"got={got} {extra}"))


# ---------------------------------------------------------------------------
# NULL B6 -- REVEAL-GATED NO-GO  (F-85 / F-89(c); s136, s140).
#   Senses in-zone at a moderate rate; on a SAFE-reveal mirror it crosses and
#   succeeds (no hazard to avoid there => own_avoid 0); on a HAZARD-reveal mirror
#   the reveal INDUCES a task-failing HALT (32-step timeout, success=False), where
#   own_blind (reveal-zeroed) would blind-cross and SUCCEED. "Half of Mode C":
#   sense->detect->halt WITHOUT the reroute. Genuinely senses causally -- the
#   sharpest near-pass. MUST be rejected (not SELECTIVE_COMPOSITE).
# ---------------------------------------------------------------------------
reveal_gated_no_go = _grade(
    {
        # safe-reveal mirror: succeeds, senses, adapt fires, but no hazard avoided
        "train_lower": _surface(success=True, total_sense=3, rate=0.375, adapt=True,
                                 own_avoided=0, hazard_avoided_vs_no_sense=0, inzone=0.5),
        # hazard-reveal mirror: FREEZES (fails); own_avoid>0 (blinding would raise
        # hazard) but success=False so it cannot be a witness
        "train_upper": _surface(success=False, total_sense=3, rate=0.375, adapt=True,
                                 own_avoided=2, hazard_avoided_vs_no_sense=2, inzone=0.5),
    },
    n=16, total_sense=6,
)
v, tc = verdict(reveal_gated_no_go)
check("B6 reveal-gated no-go NOT a C1 witness", v, {"SELECTIVE_COMPOSITE"}, "(expect SENSES_NOT_CAUSAL)")
CHECKS.append(("B6 verdict == SENSES_NOT_CAUSAL", v == "SENSES_NOT_CAUSAL", f"got={v}"))
CHECKS.append(("B6 positionally-induced FLAGGED (senses-in-zone, zero causal witness)",
               tc["positionally_induced"]["suspected"] is True,
               f"suspected={tc['positionally_induced']['suspected']}"))

# ---------------------------------------------------------------------------
# NULL B7 -- ARMED HAZARD-FREE HALF-SUCCESS LIMBO  (F-82; s131). Saturated
#   always-sense (rate == 1 pooled) with a per-layout success SPLIT (lower 1.0 /
#   upper 0.0). Degenerate-guard territory for C1; the half-success split is the
#   C4-null (per-surface Wilson, not pooled). MUST be rejected as C1.
# ---------------------------------------------------------------------------
half_success_always_sense = _grade(
    {
        "train_lower": _surface(success=True, total_sense=8, rate=1.0, adapt=True,
                                own_avoided=1, inzone=1.0),
        "train_upper": _surface(success=False, total_sense=8, rate=1.0, adapt=False,
                                own_avoided=0, inzone=1.0),
    },
    n=16, total_sense=16,
)
v, _ = verdict(half_success_always_sense)
check("B7 saturated-always half-success NOT a C1 witness", v, {"SELECTIVE_COMPOSITE"})
CHECKS.append(("B7 verdict == FAIL_ALWAYS_SENSE (degenerate guard)", v == "FAIL_ALWAYS_SENSE", f"got={v}"))

# NEAR-always variant (rate 0.9, not exactly 1) -- the CEIL conjunct (AM-2) must
# still reject even a genuine causal chain on the succeeding layout.
near_always_half = _grade(
    {
        "train_lower": _surface(success=True, total_sense=9, rate=0.9, adapt=True,
                                own_avoided=1, inzone=0.95),
        "train_upper": _surface(success=False, total_sense=9, rate=0.9, adapt=False,
                                own_avoided=0, inzone=0.95),
    },
    n=20, total_sense=18,
)
v, _ = verdict(near_always_half)
check("B7' near-always (rate>CEIL) causal-but-not-selective, NOT a witness", v, {"SELECTIVE_COMPOSITE"})
CHECKS.append(("B7' verdict == CAUSAL_BUT_NOT_SELECTIVE (AM-2 ceiling)", v == "CAUSAL_BUT_NOT_SELECTIVE", f"got={v}"))

# ---------------------------------------------------------------------------
# NULL A6 -- BLIND-CROSSING-AT-CEILING / decorative constraint (F-89(a); s138).
#   Never senses; blind-crosses at ceiling-pinned lambda=5.0. For C1: degenerate
#   FAIL_NEVER_SENSE. (Its stability-tail rejection is R1's joint success/hazard
#   read -- hazard axis fails; a marginal "high lambda" read is uninformative.)
# ---------------------------------------------------------------------------
blind_at_ceiling = _grade(
    {
        "train_lower": _surface(success=True, total_sense=0, rate=0.0, adapt=False, own_avoided=0),
        "train_upper": _surface(success=True, total_sense=0, rate=0.0, adapt=False, own_avoided=0),
    },
    n=16, total_sense=0,
)
v, _ = verdict(blind_at_ceiling)
check("A6 blind-at-ceiling NOT a C1 witness", v, {"SELECTIVE_COMPOSITE"})
CHECKS.append(("A6 verdict == FAIL_NEVER_SENSE (degenerate guard)", v == "FAIL_NEVER_SENSE", f"got={v}"))

# ---------------------------------------------------------------------------
# POSITIVE CONTROL -- a genuine reveal-dependent selective witness MUST still pass
#   (the enumeration must not have broken the pass path): success + sparse-sense +
#   adapt + own-avoid on a succeeding surface.
# ---------------------------------------------------------------------------
genuine_witness = _grade(
    {
        "train_upper": _surface(success=True, total_sense=2, rate=0.2, adapt=True,
                                own_avoided=2, hazard_avoided_vs_no_sense=2, inzone=0.4),
        "train_lower": _surface(success=True, total_sense=0, rate=0.0, adapt=False, own_avoided=0),
    },
    n=16, total_sense=2,
)
v, tc = verdict(genuine_witness)
CHECKS.append(("POS genuine witness == SELECTIVE_COMPOSITE", v == "SELECTIVE_COMPOSITE", f"got={v}"))
CHECKS.append(("POS genuine witness NOT positionally-induced (causal firewall)",
               tc["positionally_induced"]["suspected"] is False,
               f"suspected={tc['positionally_induced']['suspected']}"))


DEFAULT_OUTPUT_RELPATH = Path("docs") / "evidence" / "r2_null_inventory_shift32.json"


def _output_path(argv: list[str]) -> Path:
    """Resolve where --json writes its record.

    The default is built from the RESOLVED repository root, never from an absolute
    literal, so this instrument writes inside the checkout it was run from and
    nowhere else. ``--out PATH`` overrides it.
    """
    if "--out" in argv:
        index = argv.index("--out")
        if index + 1 >= len(argv):
            raise SystemExit("--out requires a path argument")
        return Path(argv[index + 1])
    return ROOT / DEFAULT_OUTPUT_RELPATH


def main() -> int:
    print("R2 degenerate-null rejection-existence checks (SHIFT 32, DRAFT groundwork)")
    print(f"  CEIL={CEIL}  PI_FLOOR={PI_FLOOR}")
    for name, ok, extra in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {extra}")
    allok = all(ok for _, ok, _ in CHECKS)
    print("ALL PASS" if allok else "FAILURE")
    if "--json" in sys.argv:
        import json
        rec = {
            "shift": 32,
            "purpose": "rejection-existence for the post-SHIFT-22 (T1-D12..T1-D15) "
            "degenerate nulls extending R2's witness/stability inventory",
            "status": "DRAFT groundwork; pins nothing; reads no held-out; grades no run",
            "selectivity_ceiling": CEIL,
            "positionally_induced_floor": PI_FLOOR,
            "all_pass": allok,
            "checks": [{"name": n, "pass": ok, "detail": e} for n, ok, e in CHECKS],
        }
        out = _output_path(sys.argv)
        out.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
