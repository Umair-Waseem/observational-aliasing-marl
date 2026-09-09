"""DR-H5 lever-K trigger: the executable historical regression (MECH-REC item 4).

Replays the committed T1-D11/T1-D12 training curves through the REAL
``_ArmedFloorState`` implementation (arm + hatch at the production values,
midband recovery trigger at the DR-H5 pinned values) and asserts the
countersigned regression target:

    { s128: NO engage,  s130: NO engage,  s131: first engage r1884,
      s129: vacuous (0 armed rounds) }

plus the byte-exact replay of every historical arm/de-arm event (the MECH-H
anchor that proves the replay consumed the same aggregates the production
driver did), the un-latched qualifying-round count (2,116 on s131 — the
ed_h5_e3 / auditor convention), and the deterministic exit-(c) exhaustion of
the un-intervened s131 engagement (r2683, rounds_used 800).

Run BEFORE the T1-D13 launch and again at STEP-0 of the T1-D13 grade:

    python docs/evidence/drh5_trigger_regression.py

Reads training-tier curves only (I-3 clean). Exit code 0 = ALL PASS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from raas_marl.final_training.stage25_driver import (  # noqa: E402
    _ArmedFloorState,
)

# Production arm/hatch values (T1-D11..T1-D13, hold-side frozen) + the DR-H5
# pinned trigger values.
ARM_KW = dict(
    arm_window_rounds=50,
    arm_success_threshold=0.8,
    arm_hazard_threshold=0.35,
    arm_sensing_threshold=0.05,
    dearm_window_rounds=200,
    dearm_success_threshold=0.1,
    constraint_warmup_rounds=300,
)
MIDBAND_KW = dict(
    midband_window_rounds=200,
    midband_success_lo=0.25,
    midband_success_hi=0.75,
    midband_containment_widen=0.10,
    midband_hazard_cap=0.05,
    midband_max_rounds=800,
)

# Expected historical arm/de-arm event rounds (registered grades; byte-exact).
EXPECTED = {
    "t1_d11_s128": {
        "events": [
            ("arm", 952), ("dearm", 2270), ("arm", 2330), ("dearm", 2543),
            ("arm", 2589), ("dearm", 2818), ("arm", 2867), ("dearm", 3057),
            ("arm", 3116),
        ],
        "first_engage": None,
        "qualifying_rounds": 0,
        "exit": None,
    },
    "t1_d12_s130": {
        "events": [("arm", 1843)],
        "first_engage": None,
        "qualifying_rounds": 0,
        "exit": None,
    },
    "t1_d12_s131": {
        "events": [("arm", 1705)],
        "first_engage": 1884,
        "qualifying_rounds": 2116,
        "exit": ("exhausted", 2683, 800),
    },
    "t1_d12_s129": {
        "events": [],
        "first_engage": None,
        "qualifying_rounds": 0,
        "exit": None,
    },
}


#: The README section that tells a reader where the data record lives. Named here as a
#: single constant so the instrument's message and the README cannot drift apart. PHASE 5
#: MUST create a heading with exactly this text; until it does, this string is a forward
#: reference and nothing else in the artifact depends on it.
ZENODO_README_SECTION = "Data availability"


def _missing_curve_message(run_id: str, path: Path) -> str:
    """Explain a missing Zenodo-class input instead of raising a bare traceback.

    This instrument cannot run from the code repository alone, and that is CORRECT
    rather than a defect: per-round training curves total roughly 650 MB across the 55
    runs, which is disqualifying for a git repository. The exclusion was a deliberate
    packaging decision. Only its presentation was wrong -- it used to surface as an
    unhandled FileNotFoundError, which reads like a bug.
    """
    relative = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)
    return (
        f"\n{'-' * 76}\n"
        f"drh5_trigger_regression: required input not found\n"
        f"    {relative}\n\n"
        f"This instrument replays run '{run_id}' round by round through the real\n"
        f"_ArmedFloorState machine, so it needs that run's per-round training curve.\n\n"
        f"training_curve.jsonl is part of the ZENODO DATA RECORD, not of this code\n"
        f"repository. The curves total roughly 650 MB across 55 runs, so the code\n"
        f"repository ships only results/experiments/<run>/RUN_MANIFEST.json.\n\n"
        f"To run this check: download the data record, place the run directory at the\n"
        f"path above, and re-run. See the \"{ZENODO_README_SECTION}\" section of\n"
        f"README.md for the record's location and layout.\n"
        f"{'-' * 76}"
    )


def load_curve(run_id: str) -> list[dict]:
    """Load one run's per-round training curve.

    Parameters
    ----------
    run_id : str
        Run directory name under ``results/experiments/``.

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
    # Deliberately raises the natural FileNotFoundError; the reader-facing explanation
    # is applied at the CLI boundary in main(), so `replay()` and `load_curve()` stay
    # usable as library functions with an ordinary exception contract.
    path = REPO / "results" / "experiments" / run_id / "training_curve.jsonl"
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda row: row["round_index"])
    return rows


def replay(run_id: str) -> dict:
    """Feed the curve aggregates through the real state machine.

    Replays a recorded run through the production ``_ArmedFloorState`` imported from
    the package, so the regression exercises the shipped controller rather than a
    reimplementation of it.

    Parameters
    ----------
    run_id : str
        Run directory name under ``results/experiments/``.

    Returns
    -------
    dict
        Replay outcome: row count, arm/de-arm events, first recovery engage and exit,
        armed-round count, qualifying-round count and the exhaustion flag.
    """

    rows = load_curve(run_id)
    state = _ArmedFloorState(**ARM_KW, **MIDBAND_KW)
    replayed_events: list[tuple[str, int]] = []
    engage_rounds: list[int] = []
    exit_record = None
    armed_rounds = 0
    seen = 0
    for row in rows:
        before = len(state.events)
        armed = state.observe_round(
            round_index=row["round_index"],
            success_rate=row["success_rate"],
            hazard_mean=row["hazard_cost_sum_mean"],
            sensing_mean=row["sensing_rate_mean"],
        )
        armed_rounds += int(armed)
        for event in state.events[before:]:
            kind = event["event"]
            if kind in ("arm", "dearm"):
                replayed_events.append((kind, event["round_index"]))
            elif kind == "recovery_engage":
                engage_rounds.append(event["round_index"])
            elif kind == "recovery_exit" and exit_record is None:
                exit_record = (
                    event["reason"],
                    event["round_index"],
                    event["rounds_used"],
                )
        seen += 1
    # Un-latched qualifying-round count (the ed_h5_e3 / auditor convention):
    # a pure re-evaluation of the containment predicate at every HISTORICAL
    # armed round, independent of the latch.
    lo, hi = MIDBAND_KW["midband_success_lo"], MIDBAND_KW["midband_success_hi"]
    widen = MIDBAND_KW["midband_containment_widen"]
    cap = MIDBAND_KW["midband_hazard_cap"]
    wrec = MIDBAND_KW["midband_window_rounds"]
    wlo, whi = max(0.0, lo - widen), min(1.0, hi + widen)
    post = [row for row in rows if row["round_index"] >= 300]
    succ = [row["success_rate"] for row in post]
    haz = [row["hazard_cost_sum_mean"] for row in post]
    armed_flags = [bool(row["lambda_floor_armed"]) for row in post]
    qualifying = 0
    for k in range(wrec - 1, len(post)):
        if not armed_flags[k]:
            continue
        window_s = succ[k - wrec + 1 : k + 1]
        window_h = haz[k - wrec + 1 : k + 1]
        if not all(wlo <= value <= whi for value in window_s):
            continue
        mean_s = sum(window_s) / wrec
        mean_h = sum(window_h) / wrec
        if mean_h <= cap and lo <= mean_s <= hi:
            qualifying += 1
    return {
        "rows": seen,
        "events": replayed_events,
        "first_engage": engage_rounds[0] if engage_rounds else None,
        "engage_count": len(engage_rounds),
        "exit": exit_record,
        "armed_rounds": armed_rounds,
        "qualifying_rounds": qualifying,
        "exhausted": state.recovery_exhausted,
    }


def main() -> int:
    failures = []
    for run_id, expected in EXPECTED.items():
        try:
            got = replay(run_id)
        except FileNotFoundError:
            # Translate the missing Zenodo-class input into an explanation at the CLI
            # boundary. `from None` suppresses the chained traceback.
            path = REPO / "results" / "experiments" / run_id / "training_curve.jsonl"
            raise SystemExit(_missing_curve_message(run_id, path)) from None
        checks = [
            ("historical arm/dearm events", got["events"], expected["events"]),
            ("first recovery engage", got["first_engage"], expected["first_engage"]),
            ("qualifying rounds (un-latched)", got["qualifying_rounds"],
             expected["qualifying_rounds"]),
            ("first recovery exit", got["exit"], expected["exit"]),
        ]
        if run_id == "t1_d12_s129":
            checks.append(("armed rounds (vacuous control)",
                           got["armed_rounds"], 0))
        run_ok = True
        for name, actual, want in checks:
            if actual != want:
                failures.append(f"{run_id}: {name}: got {actual!r}, "
                                f"expected {want!r}")
                run_ok = False
        status = "PASS" if run_ok else "FAIL"
        print(f"[{status}] {run_id}: rows={got['rows']} "
              f"events={got['events']} first_engage={got['first_engage']} "
              f"qualifying={got['qualifying_rounds']} exit={got['exit']} "
              f"armed_rounds={got['armed_rounds']}")
    if failures:
        print("\nREGRESSION FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nDR-H5 TRIGGER REGRESSION: ALL PASS "
          "({s128: no engage, s130: no engage, s131: r1884, s129: vacuous})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
