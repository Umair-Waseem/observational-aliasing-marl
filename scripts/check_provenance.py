#!/usr/bin/env python3
"""Provenance gate: prove every grading instrument loads its code from THIS repository.

WHY THIS EXISTS
---------------
Three of the five grading instruments used to bootstrap themselves off a hardcoded
absolute path naming the authors' private research tree. Two did it unconditionally,
at ``sys.path[0]`` and ``[1]``. On the machine where that tree existed the effect was
not a crash but something worse: the instruments loaded ``raas_marl`` -- and the
sibling harness -- from the PRIVATE tree even when executed from inside a clone of
this repository. A green instrument run was therefore not evidence about the clone.
It was evidence about a directory the reader does not have.

Those paths are gone. This script is the standing check that they stay gone, and that
no equivalent leak is introduced later. It is deliberately a positive test of
provenance rather than a negative grep for one string: a grep for the old literal
would pass the moment somebody typed a different absolute path.

WHAT IT DOES
------------
For each instrument, in a SUBPROCESS whose working directory is this repository root:

  1. load the instrument as a module, by file path, executing its own bootstrap;
  2. walk ``sys.modules`` and collect every non-stdlib module that has a ``__file__``;
  3. resolve each of those paths;
  4. classify each as

       ARTIFACT      -- inside this repository (what artifact-owned code must be)
       SITE-PACKAGES -- inside a site-packages/dist-packages directory; legitimate for
                        a declared dependency (torch is the only runtime dependency
                        this artifact declares)
       ELSEWHERE     -- anything else. This is the failure the script exists to catch.
       RELATIVE      -- ``__file__`` is not an absolute path, so it cannot be located
                        at all. torch sets exactly this on its ``torch.ops`` and
                        ``torch.classes`` pseudo-modules (``__file__ == '_ops.py'``).

FAILURE CONDITIONS
------------------
  * any module resolves ELSEWHERE; or
  * any ARTIFACT-OWNED module -- ``raas_marl*`` or one of the instrument module names
    -- resolves anywhere other than inside this repository, RELATIVE included.

The second condition is the sharp one. A private copy of ``raas_marl`` installed into
site-packages would satisfy the first check and still mean the instrument never touched
this repository's source, so owned names are held to the stricter standard.

TWO CLASSIFICATION TRAPS, BOTH HIT AND BOTH CLOSED
--------------------------------------------------
The first draft of this script passed for two wrong reasons, and the fixes are the
reason the RELATIVE class and the narrow site-root rule exist:

  * ``torch.ops.__file__`` and ``torch.classes.__file__`` are the bare strings
    ``'_ops.py'`` and ``'_classes.py'``. Running ``os.path.abspath`` on them resolves
    them against the CURRENT WORKING DIRECTORY -- which this script deliberately sets
    to the artifact root -- inventing ``<artifact>/_ops.py`` and reporting two
    site-packages modules as ARTIFACT. Relative ``__file__`` values are now classified
    RELATIVE and never resolved against the cwd.
  * ``site.getsitepackages()`` returns ``sys.prefix`` ITSELF as its first entry, so
    treating its output as the site-root list classified the entire Python
    installation -- ``DLLs/_lzma.pyd`` and friends -- as SITE-PACKAGES. Site roots are
    now only directories actually named ``site-packages`` or ``dist-packages``, and the
    interpreter's prefix directories are treated as standard library.

Neither bug could have produced a false FAILURE, only a false PASS, which is the
direction that matters for a provenance gate.

USAGE
-----
    python scripts/check_provenance.py            # run the gate, print the table
    python scripts/check_provenance.py --json     # same, plus a machine-readable dump
    python scripts/check_provenance.py --probe P  # internal: the subprocess half

Exit code 0 = every row resolves where it should. Non-zero = a provenance leak.
"""

from __future__ import annotations

import json
import os
import site
import subprocess
import sys
import sysconfig
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]

#: The five grading instruments, in the order the shipping manifest lists them.
INSTRUMENTS: tuple[str, ...] = (
    "docs/evidence/held2_bar.py",
    "docs/evidence/c1_selectivity_harness.py",
    "docs/evidence/drh5_trigger_regression.py",
    "docs/evidence/r2_null_rejection_check.py",
    "docs/evidence/c1_tightened_adversarial_test.py",
)

#: Module-name prefixes this artifact owns. These must resolve inside ARTIFACT_ROOT.
OWNED_PREFIXES: tuple[str, ...] = ("raas_marl",) + tuple(
    Path(relpath).stem for relpath in INSTRUMENTS
)


def _normalised(path: str | Path) -> str:
    """Case- and separator-normalised absolute path, for comparison on any platform."""
    return os.path.normcase(os.path.abspath(str(path)))


def _stdlib_roots() -> list[str]:
    """Directories whose contents count as the standard library, not as artifact code.

    Includes the interpreter prefixes, because extension modules live in
    ``<prefix>/DLLs`` on Windows and ``<prefix>/lib-dynload`` on POSIX -- neither of
    which is under ``sysconfig``'s ``stdlib`` path. Site-packages sits UNDER a prefix,
    so it is subtracted back out by the caller, which tests site roots first.
    """
    roots: list[str] = []
    for key in ("stdlib", "platstdlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            roots.append(_normalised(value))
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        if prefix:
            roots.append(_normalised(prefix))
    return sorted(set(roots))


def _site_roots() -> list[str]:
    """Directories genuinely named site-packages / dist-packages.

    Deliberately NOT ``site.getsitepackages()`` unfiltered: its first entry is
    ``sys.prefix`` itself, which would classify the entire Python installation as
    installed packages and hide anything dropped into the interpreter directory.
    """
    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:  # pragma: no cover - virtualenvs without getsitepackages
        pass
    try:
        user_site = site.getusersitepackages()
    except AttributeError:  # pragma: no cover
        user_site = None
    if user_site:
        candidates.append(user_site)
    candidates.extend(sys.path)

    roots: list[str] = []
    for entry in candidates:
        if not entry:
            continue
        normalised = _normalised(entry)
        if os.path.basename(normalised) in ("site-packages", "dist-packages"):
            roots.append(normalised)
    return sorted(set(roots))


def _within(path: str, roots: list[str]) -> bool:
    """True when *path* sits inside any directory in *roots*."""
    return any(path == root or path.startswith(root + os.sep) for root in roots)


# --------------------------------------------------------------------------- probe


def _probe(instrument: Path) -> int:
    """Subprocess half: import *instrument* and dump what got loaded, as JSON.

    Runs in a fresh interpreter so one instrument's imports cannot mask another's.
    """
    import importlib.util

    stdlib = _stdlib_roots()
    site_dirs = _site_roots()
    before = set(sys.modules)

    spec = importlib.util.spec_from_file_location(instrument.stem, instrument)
    if spec is None or spec.loader is None:  # pragma: no cover - unreadable file
        print(json.dumps({"error": f"cannot load {instrument}"}))
        return 2

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    # Present the instrument with its own argv, so argv-sniffing instruments
    # (r2_null_rejection_check reads sys.argv for --json) see a plain invocation.
    saved_argv = sys.argv
    sys.argv = [str(instrument)]
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:  # noqa: BLE001 - report, never swallow
        print(
            json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "sys_path_head": [str(p) for p in sys.path[:3]],
                }
            )
        )
        return 3
    finally:
        sys.argv = saved_argv

    loaded: list[dict[str, str]] = []
    for name, mod in sorted(sys.modules.items()):
        if mod is None:
            continue
        owned = is_owned(name)
        origin = getattr(mod, "__file__", None)
        if not origin:
            # Builtin, frozen or namespace package -- normally uninteresting. But an
            # ARTIFACT-OWNED name with no __file__ is the opposite of uninteresting: it
            # is exactly how a module loaded from somewhere else would become invisible
            # to this gate. Skipping it unconditionally was a hole, reproduced by an
            # adversarial audit with a fixture that loaded a private `raas_marl` and
            # registered it under the owned name with __file__ = None -- the private
            # tree did not appear in a single row. Owned names are now reported with a
            # sentinel origin so the driver can fail them.
            if owned:
                loaded.append(
                    {
                        "module": name,
                        "path": "<no __file__>",
                        "absolute": "False",
                        "newly_imported": str(name not in before),
                    }
                )
            continue

        # A non-absolute __file__ cannot be located. Do NOT abspath() it: this process
        # runs with cwd == the artifact root, so abspath would manufacture a path that
        # looks like it lives here. Report it as-is and let the driver flag it.
        if not os.path.isabs(origin):
            loaded.append(
                {
                    "module": name,
                    "path": origin,
                    "absolute": "False",
                    "newly_imported": str(name not in before),
                }
            )
            continue

        resolved = _normalised(origin)
        # Site-packages sits UNDER an interpreter prefix, so it must be tested first or
        # every installed dependency would be misfiled as standard library.
        if not owned and not _within(resolved, site_dirs) and _within(resolved, stdlib):
            continue  # standard library: not this artifact's provenance problem
        loaded.append(
            {
                "module": name,
                "path": str(Path(origin).resolve()),
                "absolute": "True",
                "newly_imported": str(name not in before),
            }
        )

    print(
        json.dumps(
            {
                "instrument": str(instrument),
                "cwd": str(Path.cwd()),
                "sys_path_head": [str(p) for p in sys.path[:3]],
                "loaded": loaded,
            }
        )
    )
    return 0


# -------------------------------------------------------------------------- driver


def classify(path: str, artifact_root: str, site_roots: list[str], absolute: bool = True) -> str:
    """Return ARTIFACT, SITE-PACKAGES, RELATIVE or ELSEWHERE for a module path.

    ``absolute=False`` short-circuits to RELATIVE: an unlocatable ``__file__`` must
    never be resolved against the current working directory, because this gate
    deliberately runs with the cwd set to the artifact root.
    """
    if not absolute:
        return "RELATIVE"
    normalised = _normalised(path)
    if _within(normalised, [artifact_root]):
        return "ARTIFACT"
    if _within(normalised, site_roots):
        return "SITE-PACKAGES"
    return "ELSEWHERE"


def is_owned(module_name: str) -> bool:
    """True when *module_name* is code this artifact ships and therefore must own."""
    root = module_name.split(".", 1)[0]
    return root in OWNED_PREFIXES


def run() -> int:
    artifact_root = _normalised(ARTIFACT_ROOT)
    site_roots = _site_roots()

    print("PROVENANCE GATE")
    print(f"  artifact root : {ARTIFACT_ROOT}")
    print(f"  interpreter   : {sys.executable}")
    print(f"  site roots    : {len(site_roots)} directory(ies)")
    print()

    rows: list[tuple[str, str, str, str]] = []
    failures: list[str] = []
    report: dict[str, object] = {"artifact_root": str(ARTIFACT_ROOT), "instruments": []}

    for relpath in INSTRUMENTS:
        instrument = ARTIFACT_ROOT / relpath
        if not instrument.is_file():
            failures.append(f"{relpath}: MISSING from the artifact")
            continue

        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe", str(instrument)],
            cwd=str(ARTIFACT_ROOT),
            capture_output=True,
            text=True,
        )
        stdout = completed.stdout.strip().splitlines()
        payload = None
        for line in reversed(stdout):  # instruments may print before we do
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            break

        if payload is None:
            failures.append(
                f"{relpath}: probe produced no JSON (exit {completed.returncode}); "
                f"stderr tail: {completed.stderr.strip()[-300:]}"
            )
            continue
        # The probe's EXIT CODE is authoritative, and is checked BEFORE trusting its
        # payload. Scanning stdout for the last parseable JSON and stopping there was a
        # hole: an adversarial audit reproduced a fixture that loaded a private
        # `raas_marl`, printed a forged clean payload, then exited non-zero -- and this
        # driver returned PASS. A probe that did not exit 0 has not established anything.
        if completed.returncode != 0:
            failures.append(
                f"{relpath}: probe exited {completed.returncode}; its output is not "
                f"trustworthy. stderr tail: {completed.stderr.strip()[-300:]}"
            )
            continue
        if "error" in payload:
            failures.append(f"{relpath}: probe failed -- {payload['error']}")
            continue

        entry: dict[str, object] = {
            "instrument": relpath,
            "cwd": payload["cwd"],
            "sys_path_head": payload["sys_path_head"],
            "modules": [],
        }
        owned_inside = 0
        for record in payload["loaded"]:
            absolute = record.get("absolute", "True") == "True"
            verdict = classify(record["path"], artifact_root, site_roots, absolute=absolute)
            inside = "yes" if verdict == "ARTIFACT" else "no"
            owned = is_owned(record["module"])
            if verdict == "ELSEWHERE":
                failures.append(
                    f"{relpath}: {record['module']} resolved OUTSIDE the artifact and "
                    f"outside site-packages -> {record['path']}"
                )
            elif owned and verdict != "ARTIFACT":
                failures.append(
                    f"{relpath}: artifact-owned module {record['module']} resolved "
                    f"{verdict} -> {record['path']}"
                )
            if owned and verdict == "ARTIFACT":
                owned_inside += 1
            rows.append((relpath, record["module"], record["path"], f"{inside} ({verdict})"))
            entry["modules"].append(
                {
                    "module": record["module"],
                    "path": record["path"],
                    "verdict": verdict,
                    "owned": owned,
                }
            )

        # NON-VACUITY. A gate that passes because it inspected nothing is worthless.
        # Every instrument must contribute at least its own module as an owned row
        # resolving inside the artifact; the four that import raas_marl contribute far
        # more. Zero owned rows means the probe silently failed to load anything.
        if owned_inside == 0:
            failures.append(
                f"{relpath}: VACUOUS -- zero artifact-owned modules were resolved, so "
                "this instrument's provenance was not actually checked"
            )
        entry["owned_inside"] = owned_inside
        report["instruments"].append(entry)  # type: ignore[union-attr]

    _print_table(rows)

    print()
    if failures:
        print(f"FAIL -- {len(failures)} provenance violation(s):")
        for failure in failures:
            print(f"  * {failure}")
    else:
        owned_rows = sum(1 for row in rows if is_owned(row[1]))
        print(
            f"PASS -- {len(rows)} module resolution(s) checked across "
            f"{len(INSTRUMENTS)} instruments; {owned_rows} artifact-owned, all inside "
            "the artifact; no module resolved ELSEWHERE."
        )

    report["failures"] = failures
    if "--json" in sys.argv:
        print()
        print(json.dumps(report, indent=2, sort_keys=True))

    return 1 if failures else 0


def _print_table(rows: list[tuple[str, str, str, str]]) -> None:
    """Print the instrument / module / path / inside-artifact table.

    Importing torch drags in ~700 third-party modules per instrument, which would bury
    the rows that carry the verdict. So every ARTIFACT, ELSEWHERE and RELATIVE row is
    printed in full, and the SITE-PACKAGES rows -- which are legitimate by construction
    -- are collapsed to one line per instrument per top-level distribution. The failure
    logic in run() still inspects every row individually; only the display is folded.
    ``--json`` emits the unfolded list.
    """
    if not rows:
        print("(no rows)")
        return

    detailed = [row for row in rows if "(SITE-PACKAGES)" not in row[3]]
    third_party: dict[tuple[str, str], int] = {}
    for instrument, module, _path, verdict in rows:
        if "(SITE-PACKAGES)" in verdict:
            key = (instrument, module.split(".", 1)[0])
            third_party[key] = third_party.get(key, 0) + 1

    headers = ("instrument", "module", "resolved path", "inside artifact?")
    display = [(Path(a).name, b, _display_path(c), d) for a, b, c, d in detailed]
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in display)) for i in range(4)
    ]
    line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(4)))
    for row in display:
        print("  ".join(row[i].ljust(widths[i]) for i in range(4)))

    if third_party:
        print()
        print("third-party (site-packages) imports, folded -- legitimate by construction:")
        for (instrument, distribution), count in sorted(third_party.items()):
            print(f"  {Path(instrument).name:<34} {distribution:<20} {count:>5} module(s)")


def _display_path(path: str) -> str:
    """Shorten a path against the artifact root so the table stays readable.

    A RELATIVE ``__file__`` is echoed verbatim -- resolving it here would re-introduce
    the very cwd-relative confusion the RELATIVE class exists to expose.
    """
    if not os.path.isabs(path):
        return f"{path}  (not absolute)"
    try:
        return "<artifact>/" + str(Path(path).resolve().relative_to(ARTIFACT_ROOT)).replace(
            os.sep, "/"
        )
    except ValueError:
        parts = Path(path).parts
        return ".../" + "/".join(parts[-3:]) if len(parts) > 3 else path


def main() -> int:
    if "--probe" in sys.argv:
        index = sys.argv.index("--probe")
        if index + 1 >= len(sys.argv):
            print(json.dumps({"error": "--probe requires a path"}))
            return 2
        return _probe(Path(sys.argv[index + 1]))
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
