"""Pytest configuration for the Segment 1 from-scratch suite.

Written fresh (NOT derived from the archived conftest). It puts ``src`` on the
import path and excludes the Phase-0 dead-zone archive from collection. The
authoritative suite lives in ``tests/test_clean.py`` and is run with:

    python -m pytest -p no:cacheprovider tests/test_clean.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The Phase-0 archive (tests/_archived/, src/_archived/) is a read-only dead zone.
# _sections/* holds the intermediate section modules that were assembled into
# test_clean.py (removed after assembly; ignored here as a safeguard).
collect_ignore_glob = ["_archived/*", "_sections/*"]
