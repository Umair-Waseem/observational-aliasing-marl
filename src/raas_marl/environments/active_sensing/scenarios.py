"""Public scenario helpers for the active-sensing grid environment.

This module is intentionally a thin convenience layer over the canonical
Stage 23-A scenario catalog in ``grid_environment.py``. It re-exports the
catalog access surface without duplicating scenario definitions.
"""

from __future__ import annotations

from raas_marl.environments.active_sensing.grid_environment import (
    Stage23Scenario,
    stage23_scenario_catalog,
)


def available_scenarios() -> tuple[str, ...]:
    """Return available scenario names in deterministic order.

    Deliberately rebuilds the full Stage 23-A catalog (four dataclasses, each
    running BFS reachability validation) on every call rather than caching a
    name-only list; ``grid_environment`` is the single source of truth for
    scenario identity and this thin layer avoids duplicating a names constant
    that could drift from the catalog (NEW-scenarios-1).
    """

    # ``dict`` keys are unique, so ``sorted`` yields a sorted, duplicate-free tuple.
    return tuple(sorted(stage23_scenario_catalog()))


def make_scenario(name: str) -> Stage23Scenario:
    """Return a validated scenario by name.

    Raises ``TypeError`` when ``name`` is not a ``str`` and ``ValueError`` when
    it is a string that names no known scenario (convention: TypeError-for-type,
    ValueError-for-value). Like :func:`available_scenarios`, the full catalog is
    rebuilt per call (NEW-scenarios-1).
    """

    if not isinstance(name, str):
        raise TypeError("scenario name must be a str")
    catalog = stage23_scenario_catalog()
    if name not in catalog:
        raise ValueError(f"unknown active-sensing scenario: {name}")
    return catalog[name]


__all__ = ["Stage23Scenario", "available_scenarios", "make_scenario", "stage23_scenario_catalog"]
