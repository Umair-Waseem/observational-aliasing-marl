"""Release instruments for the ICET 2026 paper "Observational Aliasing Bounds
Generalization in Constrained Active-Sensing Multi-Agent Reinforcement Learning".

Three standalone modules that recompute the paper's reported numbers from the
records shipped in this repository:

    reroute_basin.py   the reroute-basin classifier   (equivariant/overshoot/freeze)
    grade_composer.py  the grade composer             (the two-of-three rule)
    statistics.py      Clopper-Pearson intervals and the Fisher exact test

and one driver that runs all three and prints the paper's Table III:

    reproduce_table3.py

Every module here is deliberately SELF-CONTAINED: standard library only. They
import nothing from ``raas_marl``, nothing from the research repository, and
neither ``torch`` nor ``scipy``. Reading a shipped JSON record is all they do,
so a reviewer can check the paper's arithmetic without installing anything.

Run them from the repository root::

    python -m instruments.reproduce_table3
"""

__all__ = ["grade_composer", "reroute_basin", "statistics"]
