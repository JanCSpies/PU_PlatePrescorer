"""Do site-specific descriptors help literature-to-JnJ transfer (subset-first evaluation)?

Compares feature spaces for the winning importance-weighted PU model under the frozen protocol:

* ``ecfp``          — substrate Morgan fingerprint (the current baseline);
* ``morfeus``       — reactive-site sterics (Sterimol/%V_bur/SASA/pyramidalisation);
* ``gxtb``          — reactive-site electronics (gap/IP/EA/dipole/charge/WBO);
* ``site``          — morfeus ⊕ g-xTB;
* ``ecfp_site``     — ecfp ⊕ site.

Each is evaluated **transductively** (U = full JnJ grid) and **inductively** (U excludes TEST
substrates) — the honest generalisation test. Site features are also compared across the three
conformer-aggregation modes (boltzmann / lowest / mean). For every arm we report the
**domain-shortcut AUC** (source-vs-grid separability): the hypothesis is that physical descriptors
are more domain-invariant than fingerprints, so a *lower* domain AUC with *held* transfer would be
the win. No JnJ labels ever enter training.

Usage:
    uv run python scripts/15_site_descriptor_eval.py [--n-estimators 40]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bhpu.adapt import DomainDiscriminator
from bhpu.chem.canonical import canonical_smiles
from bhpu.data import schema
from bhpu.eval.metrics import feasibility_metrics
from bhpu.eval.protocol import FrozenEvalProtocol, _substrate_key
from bhpu.models.boosting import CatBoostFeasibilityModel
from bhpu.pu.bagging import BaggingPULearner

FEATURES_DIR = Path("datasets/processed/features")
ARTIFACTS_DIR = Path("artifacts")
SITE_METHODS = ("morfeus", "gxtb")
MODES = ("boltzmann", "lowest", "mean")


def _load_ecfp(name: str) -> np.ndarray:
    """Load a cached ECFP matrix (row-aligned to its frame)."""
    with np.load(FEATURES_DIR / f"{name}__ecfp2048.npz") as data:
        return np.asarray(data["X"], dtype=np.float32)


def _load_site_cache(method: str, role: str, mode: str) -> dict[str, np.ndarray]:
    """Load a ``{method}_{role}_{mode}`` descriptor cache as ``canonical SMILES -> vector``."""
    frame = pd.read_parquet(FEATURES_DIR / f"{method}_{role}_{mode}.parquet")
    columns = [c for c in frame.columns if c != "smiles"]
    values = frame[columns].to_numpy(dtype=np.float32)
    return {smiles: values[i] for i, smiles in enumerate(frame["smiles"])}


def _site_matrix(
    frame: pd.DataFrame, method: str, mode: str, width_aryl: int, width_amine: int
) -> np.ndarray:
    """Build an aryl⊕amine site-descriptor matrix for a frame from the caches (zeros if missing)."""
    aryl_cache = _load_site_cache(method, "aryl", mode)
    amine_cache = _load_site_cache(method, "amine", mode)

    def block(smiles: object, cache: dict[str, np.ndarray], width: int) -> np.ndarray:
        canonical = canonical_smiles(smiles) if isinstance(smiles, str) else None
        vector = cache.get(canonical) if canonical is not None else None
        return vector if vector is not None else np.zeros(width, dtype=np.float32)

    aryl = np.vstack([block(s, aryl_cache, width_aryl) for s in frame[schema.ARYL_SMILES]])
    amine = np.vstack([block(s, amine_cache, width_amine) for s in frame[schema.AMINE_SMILES]])
    return np.hstack([aryl, amine]).astype(np.float32)


def _site_widths(method: str) -> tuple[int, int]:
    """Descriptor widths (aryl, amine) for a site method, from its boltzmann cache."""
    aryl = pd.read_parquet(FEATURES_DIR / f"{method}_aryl_boltzmann.parquet")
    amine = pd.read_parquet(FEATURES_DIR / f"{method}_amine_boltzmann.parquet")
    return aryl.shape[1] - 1, amine.shape[1] - 1


def _inductive_grid_mask(
    grid: pd.DataFrame, jnj: pd.DataFrame, test_index: np.ndarray
) -> np.ndarray:
    """Boolean mask of grid rows that reuse **no** TEST substrate (inductive check)."""
    test = jnj.loc[test_index]
    test_keys = {
        k
        for col in (schema.ARYL_SMILES, schema.AMINE_SMILES)
        for s in test[col]
        if (k := _substrate_key(s)) is not None
    }
    aryl = [_substrate_key(s) for s in grid[schema.ARYL_SMILES]]
    amine = [_substrate_key(s) for s in grid[schema.AMINE_SMILES]]
    return np.array(
        [a not in test_keys and n not in test_keys for a, n in zip(aryl, amine, strict=True)]
    )


class FeatureSpace:
    """Assembles P/U/val/test matrices for one feature configuration (ecfp/site/combo, mode)."""

    def __init__(self, lit: pd.DataFrame, grid: pd.DataFrame, jnj: pd.DataFrame) -> None:
        """Cache the row-aligned frames and their ECFP matrices."""
        self.lit, self.grid, self.jnj = lit, grid, jnj
        self.ecfp = {
            "lit": _load_ecfp("literature_clean"),
            "grid": _load_ecfp("unlabeled_grid"),
            "jnj": _load_ecfp("test_jnj"),
        }

    def matrix(self, which: str, feature: str, mode: str) -> np.ndarray:
        """Return the feature matrix for a frame (``lit``/``grid``/``jnj``) and feature spec."""
        frame = getattr(self, which)
        blocks: list[np.ndarray] = []
        if feature in ("ecfp", "ecfp_site"):
            blocks.append(self.ecfp[which])
        if feature in ("morfeus", "site", "ecfp_site"):
            wa, wm = _site_widths("morfeus")
            blocks.append(_site_matrix(frame, "morfeus", mode, wa, wm))
        if feature in ("gxtb", "site", "ecfp_site"):
            wa, wm = _site_widths("gxtb")
            blocks.append(_site_matrix(frame, "gxtb", mode, wa, wm))
        return np.hstack(blocks).astype(np.float32)


def _evaluate(
    space: FeatureSpace,
    feature: str,
    mode: str,
    *,
    protocol: FrozenEvalProtocol,
    pos_mask: np.ndarray,
    grid_mask: np.ndarray,
    n_estimators: int,
    seed: int,
) -> dict[str, object]:
    """Fit importance-weighted PU-bagging for one feature/mode/grid and score VAL/TEST."""
    x_lit = space.matrix("lit", feature, mode)
    x_grid_full = space.matrix("grid", feature, mode)
    x_jnj = space.matrix("jnj", feature, mode)
    x_grid = x_grid_full[grid_mask]

    x_pos = x_lit[pos_mask]
    discriminator = DomainDiscriminator(random_state=seed).fit(x_lit, x_grid)
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

    val = feasibility_metrics(
        protocol.labels("val"), learner.predict_proba(x_jnj[protocol.val_index])
    )
    test = feasibility_metrics(
        protocol.labels("test"), learner.predict_proba(x_jnj[protocol.test_index])
    )
    return {
        "val_auc": round(val.roc_auc, 3),
        "val_auprc": round(val.auprc, 3),
        "test_auc": round(test.roc_auc, 3),
        "test_auprc": round(test.auprc, 3),
        "domain_auc": round(discriminator.domain_auc_, 3),
        "n_features": int(features.shape[1]),
    }


def main() -> None:
    """Entry point: run the feature × mode × grid sweep and print/save the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lit = pd.read_parquet(FEATURES_DIR / "literature_clean.parquet")
    grid = pd.read_parquet(FEATURES_DIR / "unlabeled_grid.parquet")
    jnj = pd.read_parquet(FEATURES_DIR / "test_jnj.parquet")
    protocol = FrozenEvalProtocol(jnj)
    pos_mask = lit[schema.LABEL].to_numpy() == 1
    space = FeatureSpace(lit, grid, jnj)

    transductive = np.ones(len(grid), dtype=bool)
    inductive = _inductive_grid_mask(grid, jnj, protocol.test_index)
    grids = {"transductive": transductive, "inductive": inductive}
    print(f"grid rows: transductive={transductive.sum()} inductive={inductive.sum()}")

    # ecfp is mode-independent; site-containing features are swept over aggregation modes.
    arms = [("ecfp", "boltzmann")] + [
        (feature, mode) for feature in ("morfeus", "gxtb", "site", "ecfp_site") for mode in MODES
    ]

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACTS_DIR / "site_descriptor_eval.json"
    results: list[dict[str, object]] = []
    header = f"\n{'feature':<10} {'mode':<10} {'grid':<12} {'dom':>5} {'VAL':>6} {'TEST':>6}"
    print(f"{header} {'TE_AP':>6}", flush=True)
    print("-" * 62, flush=True)
    for feature, mode in arms:
        for grid_name, grid_mask in grids.items():
            metrics = _evaluate(
                space,
                feature,
                mode,
                protocol=protocol,
                pos_mask=pos_mask,
                grid_mask=grid_mask,
                n_estimators=args.n_estimators,
                seed=args.seed,
            )
            row = {"feature": feature, "mode": mode, "grid": grid_name, **metrics}
            results.append(row)
            print(
                f"{feature:<10} {mode:<10} {grid_name:<12} {metrics['domain_auc']:>5} "
                f"{metrics['val_auc']:>6} {metrics['test_auc']:>6} {metrics['test_auprc']:>6}",
                flush=True,
            )
            # Write after every arm so a preemption keeps the results computed so far.
            out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
