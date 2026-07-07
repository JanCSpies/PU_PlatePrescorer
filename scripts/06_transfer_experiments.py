"""Run the honest literature→JnJ transfer arm matrix and report VAL/TEST metrics.

Trains only on literature/external data (no JnJ labels) and scores the frozen substrate-disjoint
JnJ VAL/TEST splits. Compares naive transfer against CORAL alignment and covariate-shift
importance weighting, for both PU (positives + unlabeled grid) and literature-supervised arms.

Usage:
    uv run python scripts/06_transfer_experiments.py [--featurizer drfp2048] [--n-estimators 60]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.data import schema
from bhpu.eval.transfer import ArmResult, TransferExperiment
from bhpu.features import ECFPFeaturizer

FEATURES_DIR = Path("datasets/processed/features")
ARTIFACTS_DIR = Path("artifacts")


def _featurize_corpus(path: Path, n_bits: int = 2048) -> np.ndarray:
    """ECFP-featurise a mined substrate-pair corpus (columns ``aryl_smiles``/``amine_smiles``).

    The mined corpus has no reaction conditions, so it is featurised with substrate ECFP (the
    common feature space shared by the cached literature/grid/JnJ ECFP matrices). Morgan hashing is
    deterministic and dataset-independent, so these bits align with the cache.
    """
    corpus = pd.read_parquet(path)
    frame = pd.DataFrame(
        {
            schema.ARYL_SMILES: corpus["aryl_smiles"],
            schema.AMINE_SMILES: corpus["amine_smiles"],
        }
    )
    return ECFPFeaturizer(n_bits=n_bits).fit_transform(frame)


def _row(result: ArmResult) -> str:
    """Format one arm as a fixed-width table row (VAL then TEST metrics)."""
    val, test = result.val, result.test
    ess = result.notes.get("ess", float("nan"))
    test_auc = f"{test.roc_auc:.3f}" if test else "  -  "
    test_ap = f"{test.auprc:.3f}" if test else "  -  "
    return (
        f"{result.name:<14} {result.family:<11} "
        f"{val.roc_auc:.3f}   {val.auprc:.3f}   "
        f"{test_auc}   {test_ap}   {ess:>7.0f}"
    )


def main() -> None:
    """Entry point: run all arms, print a ranked table, persist JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--featurizer", default="drfp2048", choices=["drfp2048", "ecfp2048"])
    parser.add_argument("--n-estimators", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extra-corpus",
        default=None,
        help="Parquet of mined positive substrate pairs to add (requires --featurizer ecfp2048).",
    )
    parser.add_argument("--tag", default="", help="Suffix for the artifacts filename.")
    args = parser.parse_args()

    extra_features = None
    if args.extra_corpus:
        if args.featurizer != "ecfp2048":
            parser.error("--extra-corpus needs ecfp2048 (mined corpus lacks conditions).")
        extra_features = _featurize_corpus(Path(args.extra_corpus))
        print(f"Loaded extra positive corpus {extra_features.shape} from {args.extra_corpus}")

    experiment = TransferExperiment(
        FEATURES_DIR,
        featurizer=args.featurizer,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        extra_positive_features=extra_features,
    )
    summary = experiment.summary()
    print("=== Transfer experiment ===")
    print(json.dumps(summary, indent=2))
    print(f"\nDomain shortcut diagnostic (source-vs-JnJ-grid AUC): {experiment.domain_auc:.3f}")
    print("  (≈1.0 → literature and JnJ are trivially separable; the 0.52 baseline explained.)\n")

    results = experiment.run(compute_test=True)

    # Model selection is on VAL only; TEST is shown for transparency but must not drive choices.
    best = max(results, key=lambda r: r.val.auprc)
    widths = [14, 11, 7, 7, 7, 7, 7]
    cols = ["arm", "family", "VAL_AUC", "VAL_AP", "TE_AUC", "TE_AP", "ESS"]
    print("  ".join(f"{c:<{w}}" for c, w in zip(cols, widths, strict=True)))
    print("-" * 72)
    for result in sorted(results, key=lambda r: r.val.auprc, reverse=True):
        marker = "  <- selected (VAL)" if result.name == best.name else ""
        print(_row(result) + marker)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    payload = {
        "summary": summary,
        "selected_on_val": best.name,
        "arms": [
            {
                "name": r.name,
                "family": r.family,
                "notes": r.notes,
                "val": r.val.model_dump(),
                "test": r.test.model_dump() if r.test else None,
            }
            for r in results
        ],
    }
    suffix = f"_{args.tag}" if args.tag else ""
    out = ARTIFACTS_DIR / f"transfer_{args.featurizer}{suffix}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
