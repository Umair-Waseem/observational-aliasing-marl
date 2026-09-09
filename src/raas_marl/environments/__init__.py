"""RAAS-MARL research-environments namespace-package marker.

Minimal package marker RETAINED-FOR-IMPORTS by the Segment 1 code-perfection run.
This file exists only so that ``raas_marl.environments`` is a regular importable
package whose in-scope subpackage ``raas_marl.environments.active_sensing`` (the
Stage 23-A ``RiskAwareActiveSensingGridEnvironment`` and its adapters) imports
cleanly. It deliberately re-exports nothing: it defines no public API and imports no
submodule, so importing ``raas_marl.environments`` never pulls PyTorch eagerly.
"""

from __future__ import annotations
