"""Final demo: train the "better PAYN" on literature+ORD and design a high-hit-rate BH plate.

Fits the :class:`DomainAdaptedPULearner` (transductive-U + CORAL + importance-weighted PU-bagging)
on the expanded literature+ORD positive corpus, reports the single final JnJ VAL/TEST metrics, then
scores the unlabeled combinatorial grid and ranks substrate pairs into a demo plate — quantifying
the hit-rate uplift over a randomly chosen plate (the plate-design value proposition).

TEST is touched exactly once here, with the config selected on VAL by scripts/06. Usage:
    uv run python scripts/08_plate_design_demo.py [--no-coral] [--no-iw] [--plate-size 96]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bhpu.data import schema
from bhpu.eval import TransferExperiment, plate_hit_rate_uplift
from bhpu.eval.transfer import load_corpus_features

FEATURES_DIR = Path("datasets/processed/features")
ARTIFACTS_DIR = Path("artifacts")
CORPUS = Path("datasets/processed/cn_corpus_ord.parquet")


def main() -> None:
    """Entry point: fit the adapted learner, report final metrics, design a demo plate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-coral", action="store_true", help="Disable CORAL alignment.")
    parser.add_argument("--no-iw", action="store_true", help="Disable importance weighting.")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--plate-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    extra = load_corpus_features(CORPUS)
    experiment = TransferExperiment(
        FEATURES_DIR,
        featurizer="ecfp2048",
        n_estimators=args.n_estimators,
        random_state=args.seed,
        extra_positive_features=extra,
    )
    print(json.dumps(experiment.summary(), indent=2))

    learner = experiment.fit_adapted(
        use_coral=not args.no_coral, use_importance_weights=not args.no_iw
    )
    metrics = experiment.evaluate(learner.predict_proba)
    print(
        f"\nAdapted learner  domain_auc={learner.domain_auc_:.3f}  "
        f"ESS={learner.effective_sample_size_:.0f}"
    )
    for split, m in metrics.items():
        print(f"  {split.upper()}: AUC={m.roc_auc:.3f} AUPRC={m.auprc:.3f} base={m.base_rate:.3f}")

    # Plate-design uplift on the (labelled) JnJ TEST split: how much better is a model-ranked plate?
    test_scores = learner.predict_proba(experiment.test_features)
    uplift = plate_hit_rate_uplift(
        experiment.test_labels,
        test_scores,
        budgets=(args.plate_size,),
        n_bootstrap=1000,
    )
    print("\nPlate-design uplift on JnJ TEST:")
    for r in uplift:
        print(
            f"  budget={r.budget}: model hit-rate={r.model_hit_rate:.3f} vs "
            f"random={r.random_hit_rate_mean:.3f}  → {r.uplift_vs_random:.2f}x"
        )

    # Rank the unlabeled grid into a demo plate (dedup to unique substrate pairs, keep max score).
    grid = pd.read_parquet(FEATURES_DIR / "unlabeled_grid.parquet")
    grid = grid.assign(score=learner.predict_proba(experiment.grid_features))
    best = (
        grid.sort_values("score", ascending=False)
        .drop_duplicates(subset=[schema.ARYL_SMILES, schema.AMINE_SMILES])
        .head(args.plate_size)
    )
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    plate_path = ARTIFACTS_DIR / "demo_plate.csv"
    best[[schema.ARYL_SMILES, schema.AMINE_SMILES, "score"]].to_csv(plate_path, index=False)

    payload = {
        "config": {"coral": not args.no_coral, "importance_weights": not args.no_iw},
        "domain_auc": learner.domain_auc_,
        "effective_sample_size": learner.effective_sample_size_,
        "val": metrics["val"].model_dump(),
        "test": metrics["test"].model_dump(),
        "plate_uplift": [
            {
                "budget": r.budget,
                "model_hit_rate": r.model_hit_rate,
                "random_hit_rate_mean": r.random_hit_rate_mean,
                "uplift_vs_random": r.uplift_vs_random,
                "enrichment_vs_base": r.enrichment_vs_base,
            }
            for r in uplift
        ],
    }
    (ARTIFACTS_DIR / "plate_design_demo.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {plate_path} and plate_design_demo.json")


if __name__ == "__main__":
    main()
