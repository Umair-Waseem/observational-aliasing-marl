"""HELD-2 — the paralysis-proof training-tail bar (SHIFT 21; ED-H3 section 6).

Replaces HELD-1's hazard-only conjuncts, which T1-D10 satisfied VACUOUSLY under
avoid-all paralysis (worst sliding-50 hazard 0.000-0.009 and occupancy 1.000 on all
three seeds while success was 0.000).  HELD-2's rule: NO conjunct may be satisfiable
by inactivity.

Per seed, over the tail window [3500, 4000):
  (a) JOINT Mode-C occupancy  >= 0.85 of rounds with
        success_rate >= 0.9  AND  hazard_cost_sum_mean <= 0.5   (jointly, per round)
  (b) TWO-SIDED relapse windows: no sliding 50-round window with
        mean success <  0.75   (relapse-to-paralysis / success collapse)
     and no sliding 50-round window with
        mean hazard  >= 0.75   (relapse-to-blind)
  (c) causal witness — graded EXTERNALLY by the tightened own-ablation composite
      (c1_selectivity_harness.py --grade-run --fork) on >= 2 of checkpoints
      r3500/r3750/r4000; NOT computed here.

Also provides MECH-CAP (tail paralysis-capture detector) and the BAR-C4 tail-100 read.

Diagnostic-only instrument (T1-tier training curves; training/readiness surfaces only;
no held-out read).  Self-validation: `python docs/evidence/held2_bar.py --validate`
checks synthetic paralysis / blind / alternator / late-relapse curves FAIL, a synthetic
Mode-C curve PASSES, and — the decisive regression — the three REAL t1_d10 curves FAIL
HELD-2(a) on the success conjunct exactly where HELED-1's hazard-only conjuncts passed
vacuously.  Grade a run set: `--grade-run <run_id> ...`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TAIL_START = 3500
TAIL_END = 4000
OCCUPANCY_SUCCESS = 0.9
OCCUPANCY_HAZARD = 0.5
OCCUPANCY_MIN_FRACTION = 0.85
RELAPSE_WINDOW = 50
RELAPSE_SUCCESS_FLOOR = 0.75
RELAPSE_HAZARD_CEIL = 0.75
MECH_CAP_SUCCESS_MAX = 0.1
MECH_CAP_HAZARD_MAX = 0.1
C4_TAIL = 100
C4_SUCCESS = 0.9
C4_HAZARD = 0.5

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS = _REPO_ROOT / "results" / "experiments"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


#: README section that tells a reader where the data record lives. Kept as a single
#: constant so this message and the README cannot drift. PHASE 5 must create a heading
#: with exactly this text.
ZENODO_README_SECTION = "Data availability"


def _missing_curve_message(run_id: str, path: Path) -> str:
    """Explain a missing Zenodo-class input instead of raising a bare traceback.

    ``--validate`` runs entirely on synthetic fixtures and already skips absent curves
    cleanly; only ``--grade-run`` needs real data. The exclusion of the curves from the
    code repository is a deliberate packaging decision (roughly 650 MB across 55 runs),
    so the correct behaviour here is an explanation, not a stack trace.
    """
    relative = path.relative_to(_REPO_ROOT).as_posix() if path.is_relative_to(_REPO_ROOT) else str(path)
    return (
        f"\n{'-' * 76}\n"
        f"held2_bar --grade-run: required input not found\n"
        f"    {relative}\n\n"
        f"Grading run '{run_id}' needs its per-round training curve; conjuncts (a) and\n"
        f"(b), MECH-CAP and BAR-C4 are all computed from it.\n\n"
        f"training_curve.jsonl is part of the ZENODO DATA RECORD, not of this code\n"
        f"repository. The curves total roughly 650 MB across 55 runs, so the code\n"
        f"repository ships only results/experiments/<run>/RUN_MANIFEST.json.\n\n"
        f"`--validate` needs no data at all and runs here as-is. To grade a real run,\n"
        f"download the data record and place the run directory at the path above. See\n"
        f"the \"{ZENODO_README_SECTION}\" section of README.md for the record's location.\n"
        f"{'-' * 76}"
    )


def load_run_curve(run_id: str) -> list[dict]:
    """Load one run's per-round training curve.

    Parameters
    ----------
    run_id : str
        Run directory name under ``results/experiments/``, e.g. ``t1_retain_s147``.

    Returns
    -------
    list[dict]
        One dict per round, sorted by ``round_index``.

    Raises
    ------
    FileNotFoundError
        If the curve is absent. ``training_curve.jsonl`` belongs to the Zenodo data
        record, not to this code repository; the CLI turns this into an explanation.
    """
    # Deliberately raises the natural FileNotFoundError rather than exiting: _validate()
    # CATCHES it to skip absent real-curve regressions, and any programmatic caller
    # should be able to handle a missing file its own way. The reader-facing explanation
    # is applied at the CLI boundary in main(), which is the only place that knows the
    # user asked for a grade rather than calling a library function.
    path = _EXPERIMENTS / run_id / "training_curve.jsonl"
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["round_index"])
    return rows


def held2_verdict(rows: list[dict]) -> dict:
    """HELD-2 conjuncts (a) and (b) plus MECH-CAP and BAR-C4 over a round-level curve.

    ``rows`` needs keys round_index, success_rate, hazard_cost_sum_mean.
    Conjunct (c) — the causal witness — is graded externally by the tightened
    own-ablation composite and is NOT part of this function.

    Parameters
    ----------
    rows : list[dict]
        Round-level curve records, as returned by :func:`load_run_curve`.

    Returns
    -------
    dict
        Conjunct outcomes and the window statistics behind them. Heterogeneous by
        design: booleans, floats and per-window lists in one mapping.

    Claims
    ------
    Applies the published thresholds registered in CLAIMS.yaml as
    setup-hold-tail-window, setup-checkpoint-r3500, setup-checkpoint-r4000,
    setup-hold-success, setup-hold-hazard, setup-anchor-per-mirror-hazard,
    setup-hold-occupancy, setup-relapse-window, setup-relapse-success-floor,
    setup-relapse-hazard-ceiling. Each is DEFINED as a module constant above and
    APPLIED here. setup-witness-two-of-three is deliberately NOT listed: its
    field_path records that this file only states the rule, and conjunct (c) is
    "NOT computed here".
    """
    tail = [r for r in rows if TAIL_START <= int(r["round_index"]) < TAIL_END]
    # AM-H12 (DR-H3 countersign): a partial/truncated curve must never grade a
    # partial tail — require the COMPLETE tail window and a curve that reached
    # the final round, with a deterministic ValueError either way.
    expected_tail = TAIL_END - TAIL_START
    if len(tail) != expected_tail:
        raise ValueError(
            f"curve tail window [{TAIL_START}, {TAIL_END}) has {len(tail)} rounds; "
            f"HELD-2 requires the complete {expected_tail}-round tail (AM-H12: "
            "partial curves are not gradable)"
        )
    max_round = max(int(r["round_index"]) for r in rows)
    if max_round < TAIL_END - 1:
        raise ValueError(
            f"curve ends at round {max_round} < {TAIL_END - 1}; HELD-2 requires a "
            "run completed through the tail window (AM-H12)"
        )
    succ = [float(r["success_rate"]) for r in tail]
    haz = [float(r["hazard_cost_sum_mean"]) for r in tail]

    # (a) joint per-round Mode-C occupancy
    joint = [s >= OCCUPANCY_SUCCESS and h <= OCCUPANCY_HAZARD for s, h in zip(succ, haz)]
    occupancy = sum(joint) / len(joint)
    a_pass = occupancy >= OCCUPANCY_MIN_FRACTION

    # (b) two-sided sliding relapse windows
    worst_success_window = None
    worst_hazard_window = None
    for i in range(0, len(tail) - RELAPSE_WINDOW + 1):
        s_mean = _mean(succ[i:i + RELAPSE_WINDOW])
        h_mean = _mean(haz[i:i + RELAPSE_WINDOW])
        if worst_success_window is None or s_mean < worst_success_window:
            worst_success_window = s_mean
        if worst_hazard_window is None or h_mean > worst_hazard_window:
            worst_hazard_window = h_mean
    b_pass = (worst_success_window is not None
              and worst_success_window >= RELAPSE_SUCCESS_FLOOR
              and worst_hazard_window < RELAPSE_HAZARD_CEIL)

    # MECH-CAP: tail paralysis-capture signature
    captured = (_mean(succ) <= MECH_CAP_SUCCESS_MAX and _mean(haz) <= MECH_CAP_HAZARD_MAX)

    # BAR-C4 tail-100 (unchanged form; already paralysis-proof on the success side)
    c4_rows = rows[-C4_TAIL:]
    c4_succ = _mean([float(r["success_rate"]) for r in c4_rows])
    c4_haz = _mean([float(r["hazard_cost_sum_mean"]) for r in c4_rows])
    c4_pass = c4_succ >= C4_SUCCESS and c4_haz <= C4_HAZARD

    return {
        "tail_round_count": len(tail),
        "held2_a_joint_occupancy": round(occupancy, 4),
        "held2_a_pass": a_pass,
        "held2_b_worst_window_success": round(worst_success_window, 4),
        "held2_b_worst_window_hazard": round(worst_hazard_window, 4),
        "held2_b_pass": b_pass,
        "held2_ab_pass": a_pass and b_pass,
        "held2_c_note": "causal witness graded externally (tightened own-ablation composite, >=2 of r3500/3750/4000)",
        "mech_cap_tail_paralysis_captured": captured,
        "bar_c4_tail100_success": round(c4_succ, 4),
        "bar_c4_tail100_hazard": round(c4_haz, 4),
        "bar_c4_pass": c4_pass,
    }


# ------------------------------------------------------------------ validation
def _synthetic(success: float, hazard: float, rounds: int = 4000) -> list[dict]:
    return [{"round_index": i, "success_rate": success, "hazard_cost_sum_mean": hazard}
            for i in range(rounds)]


def _validate() -> int:
    failures: list[str] = []
    skipped: list[str] = []

    def check(name: str, condition: bool) -> None:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            failures.append(name)

    # 1. paralysis (avoid-all): success 0, hazard 0 -> HELD-1's hazard-only conjuncts
    #    would pass; HELD-2 must FAIL on (a) and (b), and MECH-CAP must flag capture.
    v = held2_verdict(_synthetic(0.0, 0.0))
    check("synthetic paralysis fails (a)", not v["held2_a_pass"])
    check("synthetic paralysis fails (b)", not v["held2_b_pass"])
    check("synthetic paralysis flagged by MECH-CAP", v["mech_cap_tail_paralysis_captured"])

    # 2. blind crossing: success 1, hazard 1 -> must FAIL both (a) and (b)
    v = held2_verdict(_synthetic(1.0, 1.0))
    check("synthetic blind fails (a)", not v["held2_a_pass"])
    check("synthetic blind fails (b)", not v["held2_b_pass"])
    check("synthetic blind not MECH-CAP", not v["mech_cap_tail_paralysis_captured"])

    # 3. 50/50 alternator (blind-cross / avoid-all round alternation):
    #    round means (0.5, 0.5) on average -> occupancy ~0 jointly, window success 0.5
    rows = [{"round_index": i,
             "success_rate": 1.0 if i % 2 == 0 else 0.0,
             "hazard_cost_sum_mean": 1.0 if i % 2 == 0 else 0.0}
            for i in range(4000)]
    v = held2_verdict(rows)
    check("synthetic alternator fails (a)", not v["held2_a_pass"])
    check("synthetic alternator fails (b)", not v["held2_b_pass"])

    # 4. genuine Mode C: success 0.98, hazard 0.05 -> PASSES (a), (b), C4
    v = held2_verdict(_synthetic(0.98, 0.05))
    check("synthetic Mode C passes (a)", v["held2_a_pass"])
    check("synthetic Mode C passes (b)", v["held2_b_pass"])
    check("synthetic Mode C passes C4", v["bar_c4_pass"])
    check("synthetic Mode C not MECH-CAP", not v["mech_cap_tail_paralysis_captured"])

    # 5. late relapse-to-blind: Mode C until r3900 then blind -> (b) must catch it
    rows = [{"round_index": i, "success_rate": 0.98 if i < 3900 else 1.0,
             "hazard_cost_sum_mean": 0.05 if i < 3900 else 1.0} for i in range(4000)]
    v = held2_verdict(rows)
    check("synthetic late blind-relapse fails (b)", not v["held2_b_pass"])

    # 6. late relapse-to-paralysis: Mode C until r3900 then avoid-all -> (b) catches
    rows = [{"round_index": i, "success_rate": 0.98 if i < 3900 else 0.0,
             "hazard_cost_sum_mean": 0.05 if i < 3900 else 0.0} for i in range(4000)]
    v = held2_verdict(rows)
    check("synthetic late paralysis-relapse fails (b)", not v["held2_b_pass"])

    # 6b. AM-H12: partial/truncated curves must raise, never grade a partial tail
    try:
        held2_verdict(_synthetic(0.98, 0.05, rounds=3600))
        check("AM-H12 truncated curve (3600 rds) raises ValueError", False)
    except ValueError:
        check("AM-H12 truncated curve (3600 rds) raises ValueError", True)
    try:
        gappy = [r for r in _synthetic(0.98, 0.05) if r["round_index"] % 7 != 0]
        held2_verdict(gappy)
        check("AM-H12 gappy tail raises ValueError", False)
    except ValueError:
        check("AM-H12 gappy tail raises ValueError", True)
    v = held2_verdict(_synthetic(0.98, 0.05))
    check("AM-H12 complete curve reports tail_round_count == 500",
          v["tail_round_count"] == TAIL_END - TAIL_START)

    # 7. THE REGRESSION THAT MOTIVATED HELD-2: the real T1-D10 curves.  HELD-1's
    #    hazard-only conjuncts passed vacuously on all three; HELD-2 must FAIL them
    #    on the success side, and MECH-CAP must flag the captured tail.
    for run_id in ("t1_d10_s123", "t1_d10_s124", "t1_d10_s125"):
        try:
            rows = load_run_curve(run_id)
        except FileNotFoundError:
            print(f"  [SKIP] real-curve regression {run_id} (curve not on disk)")
            skipped.append(run_id)
            continue
        v = held2_verdict(rows)
        check(f"real {run_id} fails (a) [HELD-1 passed vacuously here]", not v["held2_a_pass"])
        check(f"real {run_id} hazard side alone would have passed (vacuity confirmed)",
              v["held2_b_worst_window_hazard"] < RELAPSE_HAZARD_CEIL)
        check(f"real {run_id} MECH-CAP flags capture", v["mech_cap_tail_paralysis_captured"])

    # DEGRADED BANNER. The three real-curve regressions above are what this module's
    # docstring calls "the decisive regression": they are the reason HELD-2 exists,
    # because HELD-1's hazard-only conjuncts passed VACUOUSLY on exactly those curves.
    # They skip when the Zenodo-class curves are absent, which is the normal state of a
    # code-only checkout. Printing a bare "ALL CHECKS PASS" in that state would tell a
    # reader -- or a CI log -- that this instrument had exercised the vacuity it exists
    # to eliminate, when it had not. Exit stays 0, because absent data is not a failure,
    # but the gap is made impossible to miss.
    if skipped:
        print(
            "\n  *** DEGRADED: {n} of 3 real-curve regressions were SKIPPED ({ids}).\n"
            "      These are the decisive HELD-1-vacuity regressions; the synthetic\n"
            "      checks alone do NOT establish that this bar rejects the paralysis\n"
            "      basin. Supply training_curve.jsonl from the data record for full\n"
            '      coverage; see the "{section}" section of README.md.'.format(
                n=len(skipped), ids=", ".join(skipped), section=ZENODO_README_SECTION
            )
        )
    verdict = "ALL CHECKS PASS" if not failures else f"{len(failures)} CHECK(S) FAILED"
    if skipped and not failures:
        verdict += f"  (DEGRADED: {len(skipped)} real-curve regression(s) skipped)"
    print(f"\n{verdict}")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--grade-run", nargs="+", metavar="RUN_ID")
    args = parser.parse_args(argv)
    if args.validate:
        return _validate()
    if args.grade_run:
        for run_id in args.grade_run:
            try:
                rows = load_run_curve(run_id)
            except FileNotFoundError as exc:
                # Translate the missing Zenodo-class input into an explanation at the
                # CLI boundary. `from None` suppresses the chained traceback: the
                # reader needs to know the data lives elsewhere, not see a stack.
                raise SystemExit(
                    _missing_curve_message(run_id, Path(exc.filename or ""))
                ) from None
            print(json.dumps({"run": run_id, **held2_verdict(rows)}, indent=1))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
