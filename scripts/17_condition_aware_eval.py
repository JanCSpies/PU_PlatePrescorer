"""Condition-aware transfer: does encoding catalyst/base/solvent beat substrate-only?

The winning model so far encodes **only the substrates** (aryl ⊕ amine ECFP). But in the JnJ set,
conditions drive **43%** of the label variance and **35%** of pairs flip outcome across the 1,137
condition sets (32 catalysts × 8 bases × 6 solvents) — the whole point of HTE. This script adds
**structural condition fingerprints** (ECFP of the catalyst/base/solvent SMILES; the catalyst SMILES
are full Pd precatalyst complexes, so metal+ligand are captured) and evaluates the condition-level
(per-row) target under the frozen protocol.

Two fixes are needed together (see LAB_JOURNAL): (1) encode conditions, and (2) rebuild the
unlabeled grid U to **span the condition space** — U = unmeasured aryl×amine pairs × conditions
sampled from the JnJ condition distribution (not the 4 fixed general conditions). Otherwise the
condition bits become a pure literature-vs-grid domain tag (why DRFP scored below random before).

Arms (IW-PU, transductive + inductive): ``substrate`` (baseline) vs ``substrate+condition``.

Usage:
    uv run python scripts/17_condition_aware_eval.py [--n-estimators 40] [--conditions-per-pair 13]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.adapt import DomainDiscriminator
from bhpu.chem.reaction import build_cn_coupling_product
from bhpu.data import schema
from bhpu.eval.metrics import feasibility_metrics
from bhpu.eval.protocol import FrozenEvalProtocol, _substrate_key
from bhpu.features import ECFPFeaturizer
from bhpu.models.boosting import CatBoostFeasibilityModel
from bhpu.pu.bagging import BaggingPULearner

PROCESSED = Path("datasets/processed")
FEATURES_DIR = PROCESSED / "features"
ARTIFACTS_DIR = Path("artifacts")
CONDITION_COLUMNS = (schema.CATALYST_SMILES, schema.BASE_SMILES, schema.SOLVENT_SMILES)
CONDITION_NAME_COLUMNS = (schema.CATALYST_NAME, schema.BASE_NAME, schema.SOLVENT_NAME)
SUBSTRATE_BITS = 2048
CONDITION_BITS = 1024


def _load_ecfp(name: str) -> np.ndarray:
    """Load a cached substrate ECFP matrix (row-aligned to its frame)."""
    with np.load(FEATURES_DIR / f"{name}__ecfp2048.npz") as data:
        return np.asarray(data["X"], dtype=np.float32)


def _condition_ecfp_map(frames: list[pd.DataFrame]) -> dict[tuple[str, str, str], np.ndarray]:
    """ECFP every unique (catalyst, base, solvent) SMILES triple across the given frames."""
    triples = pd.concat([f[list(CONDITION_COLUMNS)] for f in frames], ignore_index=True)
    triples = triples.dropna().drop_duplicates().reset_index(drop=True)
    featurizer = ECFPFeaturizer(substrate_columns=CONDITION_COLUMNS, n_bits=CONDITION_BITS)
    matrix = featurizer.fit_transform(triples).astype(np.float32)
    return {
        (str(r[0]), str(r[1]), str(r[2])): matrix[i]
        for i, r in enumerate(triples[list(CONDITION_COLUMNS)].to_numpy())
    }


def _condition_block(
    frame: pd.DataFrame, cmap: dict[tuple[str, str, str], np.ndarray]
) -> np.ndarray:
    """Assemble the per-row condition ECFP block from the cached triple map (zeros if missing)."""
    width = CONDITION_BITS * len(CONDITION_COLUMNS)
    rows = []
    for cat, base, solv in frame[list(CONDITION_COLUMNS)].to_numpy():
        vector = cmap.get((str(cat), str(base), str(solv)))
        rows.append(vector if vector is not None else np.zeros(width, dtype=np.float32))
    return np.vstack(rows).astype(np.float32)


def _substrate_ecfp(frame: pd.DataFrame) -> np.ndarray:
    """Compute substrate (aryl⊕amine) ECFP for an arbitrary frame (deduped by pair, broadcast)."""
    pairs = (
        frame[[schema.ARYL_SMILES, schema.AMINE_SMILES]].drop_duplicates().reset_index(drop=True)
    )
    featurizer = ECFPFeaturizer(n_bits=SUBSTRATE_BITS)
    matrix = featurizer.fit_transform(pairs).astype(np.float32)
    index = {
        (a, n): i
        for i, (a, n) in enumerate(pairs[[schema.ARYL_SMILES, schema.AMINE_SMILES]].to_numpy())
    }
    order = [index[(a, n)] for a, n in frame[[schema.ARYL_SMILES, schema.AMINE_SMILES]].to_numpy()]
    return matrix[order]


def _build_condition_grid(
    jnj: pd.DataFrame, *, conditions_per_pair: int, seed: int
) -> pd.DataFrame:
    """Unmeasured aryl×amine pairs × conditions sampled from the JnJ condition distribution."""
    aryls = sorted(jnj[schema.ARYL_SMILES].dropna().unique())
    amines = sorted(jnj[schema.AMINE_SMILES].dropna().unique())
    measured = set(zip(jnj[schema.ARYL_SMILES], jnj[schema.AMINE_SMILES], strict=True))

    condition_cols = list(CONDITION_NAME_COLUMNS) + list(CONDITION_COLUMNS)
    conditions = jnj[condition_cols].dropna().reset_index(drop=True)
    counts = conditions.groupby(condition_cols).size()
    unique_conditions = counts.index.to_frame(index=False)
    weights = (counts / counts.sum()).to_numpy()

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for aryl in aryls:
        for amine in amines:
            if (aryl, amine) in measured:
                continue
            product = build_cn_coupling_product(aryl, amine)
            n = min(conditions_per_pair, len(unique_conditions))
            picks = rng.choice(len(unique_conditions), size=n, replace=False, p=weights)
            for idx in picks:
                row = unique_conditions.iloc[idx]
                records.append(
                    {
                        schema.ARYL_SMILES: aryl,
                        schema.AMINE_SMILES: amine,
                        schema.PRODUCT_SMILES: product,
                        **{c: row[c] for c in condition_cols},
                    }
                )
    return pd.DataFrame.from_records(records)


def _features(
    frame: pd.DataFrame,
    substrate: np.ndarray,
    cmap: dict[tuple[str, str, str], np.ndarray],
    *,
    with_conditions: bool,
) -> np.ndarray:
    """Assemble the feature matrix: substrate ECFP, optionally ⊕ condition ECFP."""
    if not with_conditions:
        return substrate
    return np.hstack([substrate, _condition_block(frame, cmap)]).astype(np.float32)


def _fit_and_score(
    x_pos: np.ndarray,
    x_grid: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    *,
    n_estimators: int,
    seed: int,
) -> dict[str, float]:
    """Importance-weighted PU-bagging fit + VAL/TEST scoring for one feature configuration."""
    discriminator = DomainDiscriminator(random_state=seed).fit(x_pos, x_grid)
    weights = discriminator.importance_weights(x_pos)
    features = np.vstack([x_pos, x_grid])
    selection = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_grid))])
    sample_weight = np.concatenate([weights, np.ones(len(x_grid))])

    def factory() -> CatBoostFeasibilityModel:
        return CatBoostFeasibilityModel(
            iterations=300, depth=6, learning_rate=0.05, random_state=seed
        )

    learner = BaggingPULearner(factory, n_estimators=n_estimators, random_state=seed)
    learner.fit(features, selection, sample_weight=sample_weight)
    val = feasibility_metrics(y_val, learner.predict_proba(x_val))
    test = feasibility_metrics(y_test, learner.predict_proba(x_test))
    return {
        "domain_auc": round(discriminator.domain_auc_, 3),
        "val_auc": round(val.roc_auc, 3),
        "test_auc": round(test.roc_auc, 3),
        "test_auprc": round(test.auprc, 3),
        "n_features": int(features.shape[1]),
    }


def main() -> None:
    """Entry point: build the condition-aware grid; compare substrate vs substrate+condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--conditions-per-pair", type=int, default=13)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lit = pd.read_parquet(FEATURES_DIR / "literature_clean.parquet")
    jnj = pd.read_parquet(FEATURES_DIR / "test_jnj.parquet")
    protocol = FrozenEvalProtocol(jnj)
    pos_mask = lit[schema.LABEL].to_numpy() == 1

    print(f"building condition-aware U ({args.conditions_per_pair} conditions/unmeasured pair) ...")
    grid = _build_condition_grid(jnj, conditions_per_pair=args.conditions_per_pair, seed=args.seed)
    print(f"  U rows: {len(grid)} (vs 4-condition grid ~26k)")

    cmap = _condition_ecfp_map([lit, jnj, grid])
    print(f"  unique condition triples fingerprinted: {len(cmap)}")

    # Substrate ECFP: cached for lit/jnj (row-aligned), computed for the new grid.
    sub_lit = _load_ecfp("literature_clean")
    sub_jnj = _load_ecfp("test_jnj")
    sub_grid = _substrate_ecfp(grid)

    # Inductive grid mask (drop rows reusing a TEST substrate).
    test = jnj.loc[protocol.test_index]
    test_keys = {
        k
        for col in (schema.ARYL_SMILES, schema.AMINE_SMILES)
        for s in test[col]
        if (k := _substrate_key(s)) is not None
    }
    grid_aryl = [_substrate_key(s) for s in grid[schema.ARYL_SMILES]]
    grid_amine = [_substrate_key(s) for s in grid[schema.AMINE_SMILES]]
    inductive_mask = np.array(
        [
            a not in test_keys and n not in test_keys
            for a, n in zip(grid_aryl, grid_amine, strict=True)
        ]
    )
    grids = {"transductive": np.ones(len(grid), dtype=bool), "inductive": inductive_mask}
    print(f"  grid rows transductive={len(grid)} inductive={int(inductive_mask.sum())}")

    y_val, y_test = protocol.labels("val"), protocol.labels("test")
    results: list[dict[str, object]] = []
    print(f"\n{'features':<20}{'grid':<13}{'dom':>6}{'VAL':>7}{'TEST':>7}{'TE_AP':>7}")
    print("-" * 60)
    for with_conditions in (False, True):
        name = "substrate+condition" if with_conditions else "substrate"
        x_pos = _features(lit, sub_lit, cmap, with_conditions=with_conditions)[pos_mask]
        x_grid_full = _features(grid, sub_grid, cmap, with_conditions=with_conditions)
        x_jnj = _features(jnj, sub_jnj, cmap, with_conditions=with_conditions)
        for grid_name, mask in grids.items():
            metrics = _fit_and_score(
                x_pos,
                x_grid_full[mask],
                x_jnj[protocol.val_index],
                x_jnj[protocol.test_index],
                y_val,
                y_test,
                n_estimators=args.n_estimators,
                seed=args.seed,
            )
            results.append({"features": name, "grid": grid_name, **metrics})
            print(
                f"{name:<20}{grid_name:<13}{metrics['domain_auc']:>6}"
                f"{metrics['val_auc']:>7}{metrics['test_auc']:>7}{metrics['test_auprc']:>7}"
            )
            ARTIFACTS_DIR.mkdir(exist_ok=True)
            (ARTIFACTS_DIR / "condition_aware_eval.json").write_text(json.dumps(results, indent=2))

    print(f"\nWrote {ARTIFACTS_DIR / 'condition_aware_eval.json'}")


if __name__ == "__main__":
    main()
