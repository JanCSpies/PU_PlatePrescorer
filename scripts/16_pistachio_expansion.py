"""Does expanding the positive corpus (ORD -> +Pistachio) improve literature->JnJ transfer?

Independent of the (negative) site-descriptor result, this is the payoff of mining Pistachio:
201k product-confirmed, JnJ-leak-free C-N pairs (vs ORD's 36k). We re-run the winning
importance-weighted PU model on ECFP with progressively larger positive corpora and report the
frozen-protocol VAL/TEST metrics:

    lit                     (literature positives only)
    lit + ORD               (open mined corpus)
    lit + ORD + Pistachio   (licensed expansion; deduped, substrate-disjoint from JnJ)

Pistachio is LICENSED: its corpus lives in the gitignored ``datasets/processed/private/`` and is
never committed. Model selection is on VAL; TEST is touched once per corpus.

Usage:
    uv run python scripts/16_pistachio_expansion.py [--n-estimators 60]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from bhpu.data import schema
from bhpu.eval import TransferExperiment
from bhpu.eval.transfer import load_corpus_features
from bhpu.features import ECFPFeaturizer

FEATURES_DIR = Path("datasets/processed/features")
ARTIFACTS_DIR = Path("artifacts")
ORD_CORPUS = Path("datasets/processed/cn_corpus_ord.parquet")
PISTACHIO_CORPUS = Path("datasets/processed/private/cn_corpus_pistachio.parquet")


def _deduped_corpus_features(paths: list[Path]) -> np.ndarray:
    """ECFP-featurise the union of mined corpora, deduped by (aryl, amine) substrate pair."""
    frames = [
        pd.read_parquet(p, columns=["aryl_smiles", "amine_smiles"]) for p in paths if p.exists()
    ]
    corpus = pd.concat(frames, ignore_index=True).drop_duplicates(["aryl_smiles", "amine_smiles"])
    frame = pd.DataFrame(
        {schema.ARYL_SMILES: corpus["aryl_smiles"], schema.AMINE_SMILES: corpus["amine_smiles"]}
    )
    features: np.ndarray = ECFPFeaturizer(n_bits=2048).fit_transform(frame).astype(np.float32)
    return features


def main() -> None:
    """Entry point: run the corpus-expansion comparison on the frozen protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpora: dict[str, np.ndarray | None] = {"lit": None}
    if ORD_CORPUS.exists():
        corpora["lit+ORD"] = load_corpus_features(ORD_CORPUS)
    if PISTACHIO_CORPUS.exists():
        corpora["lit+ORD+Pistachio"] = _deduped_corpus_features([ORD_CORPUS, PISTACHIO_CORPUS])

    results: list[dict[str, object]] = []
    print(f"{'corpus':<20}{'n_pos':>9}{'domAUC':>8}{'VAL':>7}{'TEST':>7}{'TE_AP':>7}")
    print("-" * 58)
    for name, extra in corpora.items():
        experiment = TransferExperiment(
            FEATURES_DIR,
            featurizer="ecfp2048",
            n_estimators=args.n_estimators,
            random_state=args.seed,
            extra_positive_features=extra,
        )
        learner = experiment.fit_adapted(use_coral=False, use_importance_weights=True)
        metrics = experiment.evaluate(learner.predict_proba)
        n_positives = cast(int, experiment.summary()["n_positives_total"])
        row = {
            "corpus": name,
            "n_positives": n_positives,
            "domain_auc": round(experiment.domain_auc, 3),
            "val_auc": round(metrics["val"].roc_auc, 3),
            "test_auc": round(metrics["test"].roc_auc, 3),
            "test_auprc": round(metrics["test"].auprc, 3),
            "effective_sample_size": round(learner.effective_sample_size_, 0),
        }
        results.append(row)
        print(
            f"{name:<20}{row['n_positives']:>9}{row['domain_auc']:>8}"
            f"{row['val_auc']:>7}{row['test_auc']:>7}{row['test_auprc']:>7}"
        )

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "pistachio_expansion.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {ARTIFACTS_DIR / 'pistachio_expansion.json'}")


if __name__ == "__main__":
    main()
