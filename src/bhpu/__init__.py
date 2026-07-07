"""bhpu — Buchwald–Hartwig C–N feasibility prediction from literature data via PU learning.

This package supports a hackathon research project (Glorius group, University of Münster)
that predicts the feasibility of Buchwald–Hartwig C–N cross-couplings using only literature
reaction data, framed as Positive-Unlabeled (PU) learning, and evaluated against a held-out
high-throughput-experimentation (HTE) dataset.

Subpackages:
    data: Dataset loading, splitting, provenance, and leakage auditing.
"""

from __future__ import annotations

__version__ = "0.1.0"
