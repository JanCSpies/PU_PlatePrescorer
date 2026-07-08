"""Modular PU-learning benchmarking pipeline for reaction feasibility prediction.

``pubench`` benchmarks positive-unlabeled (PU) learning algorithms on the condition-aware
Buchwald–Hartwig datasets assembled into ``final_datasets/`` (P / U / val / test), under a
realistic deployment setting where the SCAR assumption is violated by reporting bias and by
distribution shift between the literature positives and the unlabeled candidate pool.

The package is config-driven and extension-point-based: featurizers and PU models self-register
by name, so a new one is added by writing one class plus a decorator, with no runner changes.
See :mod:`pubench.runner.experiment` for the end-to-end entry point.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
