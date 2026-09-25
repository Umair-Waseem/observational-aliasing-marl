# Security

This repository is the research artifact of an IEEE ICET 2026 paper. `requirements.txt` pins the environment in which the reported runs were made, not the newest safe versions.

## Pinned versions

`requirements.txt` pins `torch==2.5.1`, the version the reported runs used. Security advisories have been published for `torch` 2.5.1. The pin stays so that the recorded environment can be rebuilt. Upgrading it would change the software the results came from.

The reproduce command, `python -m instruments.reproduce_table3`, uses the Python standard library only and needs no install.

## Safe use

* Install into a virtual environment, not into a system Python.
* With this environment, load only the policy checkpoints shipped in `results/experiments/`. `results/checkpoints_manifest.json` lists their SHA-256 values. Do not load `.pt` files from other sources.
* For new work that does not need the recorded environment, use a current PyTorch release.

## Reporting a problem

Report a security problem privately, through "Report a vulnerability" on this repository's Security tab. Do not open a public issue for it.
