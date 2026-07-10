"""Reaction mining for the dataset-switch positives corpora (amide coupling).

Mirrors the Buchwald-Hartwig miner in :mod:`bhpu.data.mining` but for amide (acid + amine ->
amide) couplings, and applies the *transductive* leakage rule this project uses for a chosen test
set: drop only positive rows whose exact (acid, amine) pair appears in Test, while keeping rows that
merely share a substrate (those substrates legitimately populate the transductive unlabeled grid U).
"""

from __future__ import annotations

from pubench.mining.amide import (
    AmideCouplingDetector,
    AmideMiner,
    MinedAmide,
    detect_amide_batch,
    enumerate_amide_products,
)

__all__ = [
    "AmideCouplingDetector",
    "AmideMiner",
    "MinedAmide",
    "detect_amide_batch",
    "enumerate_amide_products",
]
