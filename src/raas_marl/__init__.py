"""RAAS-MARL package root.

Minimal package marker RETAINED-FOR-IMPORTS by the Segment 1 code-perfection run.
The original ``raas_marl/__init__.py`` eagerly re-exported the pre-restart modules
(``ablations``, ``active_sensing``, ``belief``, ``final_evaluation``, ...), all of
which were archived to ``src/_archived/raas_marl/`` in Phase 0. That import surface no
longer exists in the active tree, so this file deliberately re-exports nothing and
exists only so that ``raas_marl`` remains a regular package whose in-scope subpackages
``raas_marl.mappo_lagrangian`` and ``raas_marl.environments`` import cleanly.
"""

from __future__ import annotations
