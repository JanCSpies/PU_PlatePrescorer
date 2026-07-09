"""The experiment driver: build the featurizer x model grid and produce the comparison table."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pubench.data import FinalDatasetsAdapter, PUDataset, Split
from pubench.eval import EvalResult, Evaluator
from pubench.features import build_featurizer
from pubench.models import LabeledArrays, build_model
from pubench.models.base import PUModel
from pubench.runner import provenance
from pubench.runner.config import ComponentSpec, ExperimentConfig

_LOGGER = logging.getLogger("pubench")


class Experiment:
    """Run every configured (featurizer, model) combination and collect a tidy results table.

    Parameters
    ----------
    config:
        The resolved :class:`ExperimentConfig`.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> pd.DataFrame:
        """Execute the benchmark and persist the comparison table + provenance.

        Returns:
        -------
        pandas.DataFrame
            One row per (model, featurizer, split) with assumption flags and all metrics.
        """
        dataset = FinalDatasetsAdapter(
            root=self.config.data.root,
            positives=self.config.data.positives,
            base_filter=self.config.data.base_filter,
        ).load()
        _LOGGER.info(
            "loaded splits: %s",
            {s.value: dataset.frame(s).n for s in dataset.available_splits},
        )
        evaluator = Evaluator(threshold_strategy=self.config.threshold_strategy)  # type: ignore[arg-type]

        rows: list[dict[str, object]] = []
        for f_spec in self.config.featurizers:
            featured = self._featurize(dataset, f_spec)
            for m_spec in self.config.models:
                rows.extend(self._run_combo(featured, f_spec, m_spec, evaluator))

        table = pd.DataFrame(rows)
        self._persist(table)
        return table

    def _featurize(self, dataset: PUDataset, spec: ComponentSpec) -> PUDataset:
        """Fit the featurizer on the P+U pool (label-free) and transform every split.

        Data-dependent reducers (e.g. :class:`ReducedECFPFeaturizer`'s frequency filter / SVD) must
        see the *unlabeled* target marginal U, not only the positives P, or they condense to P's
        region and miss the JnJ substrate space. U carries no labels, so fitting on P+U introduces
        no leakage and is consistent with the transductive protocol; stateless fingerprint
        featurizers are unaffected.
        """
        _LOGGER.info("featurizing with %s", spec.name)
        featurizer = build_featurizer(spec.name, spec.params)
        fit_splits = [s for s in (Split.P, Split.U) if s in dataset.available_splits]
        fit_records = pd.concat(
            [dataset.records(s) for s in fit_splits], ignore_index=True, sort=False
        )
        featurizer.fit(fit_records)
        matrices = {
            split: featurizer.transform(dataset.records(split))
            for split in dataset.available_splits
        }
        return dataset.with_features(matrices)

    def _run_combo(
        self,
        dataset: PUDataset,
        f_spec: ComponentSpec,
        m_spec: ComponentSpec,
        evaluator: Evaluator,
    ) -> list[dict[str, object]]:
        """Fit one model on the featurised data and evaluate it on val + test."""
        _LOGGER.info("fitting %s x %s", f_spec.name, m_spec.name)
        model = build_model(m_spec.name, m_spec.params, seed=self.config.seed)
        val_y = _require(dataset.labels(Split.VAL), "val labels")
        test_y = _require(dataset.labels(Split.TEST), "test labels")
        model.fit(
            dataset.matrix(Split.P),
            dataset.matrix(Split.U),
            val=LabeledArrays(dataset.matrix(Split.VAL), val_y),
        )
        val_scores = model.predict_proba(dataset.matrix(Split.VAL))
        test_scores = model.predict_proba(dataset.matrix(Split.TEST))

        results = {
            Split.VAL: evaluator.evaluate(val_scores, val_y, val_scores=val_scores, val_y=val_y),
            Split.TEST: evaluator.evaluate(test_scores, test_y, val_scores=val_scores, val_y=val_y),
        }
        return [
            self._row(model, f_spec, m_spec, split, result) for split, result in results.items()
        ]

    def _row(
        self,
        model: PUModel,
        f_spec: ComponentSpec,
        m_spec: ComponentSpec,
        split: Split,
        result: EvalResult,
    ) -> dict[str, object]:
        """Assemble one tidy results row (identity + assumption flags + metrics)."""
        row: dict[str, object] = {
            "model": m_spec.display,
            "featurizer": f_spec.display,
            "split": split.value,
            "positives": self.config.data.positives,
            "seed": self.config.seed,
            "estimated_prior": getattr(model, "prior_", None),
            "needs_prior": model.requires.needs_prior,
            "needs_propensity": model.requires.needs_propensity,
            "assumes_fixed_negatives": model.requires.assumes_fixed_negatives,
            "handles_dd_shift": model.requires.handles_dd_shift,
        }
        row.update(result.flat())
        return row

    def _persist(self, table: pd.DataFrame) -> None:
        """Write the comparison table, resolved config, and provenance to ``out_dir``."""
        import yaml

        out_dir = self.config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out_dir / "comparison_table.parquet", index=False)
        table.to_csv(out_dir / "comparison_table.csv", index=False)
        (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(self.config.to_dict()))
        provenance.write_json(
            out_dir / "provenance.json", provenance.provenance_record(self.config.to_dict())
        )
        _LOGGER.info("wrote comparison table (%d rows) to %s", len(table), out_dir)


def _require(array: np.ndarray | None, what: str) -> np.ndarray:
    """Return ``array`` or raise if it is ``None`` (a labelled split is required)."""
    if array is None:
        raise ValueError(f"Expected {what} but the split is unlabelled.")
    return array
