"""Typed, YAML-loadable experiment configuration (no hardcoded paths or hyperparameters)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the pre-sampled splits live and which positives corpus to use.

    Attributes:
    ----------
    root:
        Directory holding the assembled ``final_datasets`` parquet files.
    positives:
        Positives-corpus key (``"merged"`` or ``"ord"``).
    """

    root: Path
    positives: str = "merged"


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """A registry-resolved component: a ``name`` plus constructor ``params``.

    Attributes
    ----------
    name:
        Registry key of the component.
    params:
        Constructor keyword arguments.
    label:
        Display name for the results table; defaults to ``name``. Give distinct labels to two
        specs that share a ``name`` but differ in ``params`` (e.g. two ECFP configurations).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    @property
    def display(self) -> str:
        """Return the results-table label (``label`` if set, else ``name``)."""
        return self.label or self.name


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Full benchmark configuration: the data, the featurizer x model grid, and run settings.

    Attributes:
    ----------
    data:
        Data source configuration.
    featurizers:
        Featurizer specs to run (each is applied to every model).
    models:
        PU-model specs to run.
    seed:
        Global random seed plumbed into every component.
    threshold_strategy:
        Threshold-selection strategy for the evaluator.
    out_dir:
        Directory for the comparison table + per-run artifacts.
    """

    data: DataConfig
    featurizers: list[ComponentSpec]
    models: list[ComponentSpec]
    seed: int = 0
    threshold_strategy: str = "val_youden"
    out_dir: Path = Path("artifacts/pubench")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Build a config from a plain (YAML-parsed) dict."""
        data = DataConfig(
            root=Path(raw["data"]["root"]),
            positives=raw["data"].get("positives", "merged"),
        )
        return cls(
            data=data,
            featurizers=[_component(c) for c in raw.get("featurizers", [])],
            models=[_component(c) for c in raw.get("models", [])],
            seed=int(raw.get("seed", 0)),
            threshold_strategy=raw.get("threshold_strategy", "val_youden"),
            out_dir=Path(raw.get("out_dir", "artifacts/pubench")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serialisable representation of the resolved config."""
        return {
            "data": {"root": str(self.data.root), "positives": self.data.positives},
            "featurizers": [
                {"name": c.name, "params": c.params, "label": c.label} for c in self.featurizers
            ],
            "models": [{"name": c.name, "params": c.params, "label": c.label} for c in self.models],
            "seed": self.seed,
            "threshold_strategy": self.threshold_strategy,
            "out_dir": str(self.out_dir),
        }


def _component(raw: dict[str, Any]) -> ComponentSpec:
    """Parse one featurizer/model spec dict into a :class:`ComponentSpec`."""
    return ComponentSpec(
        name=raw["name"], params=dict(raw.get("params", {})), label=raw.get("label")
    )


def load_config(path: Path) -> ExperimentConfig:
    """Load and parse an experiment config from a YAML file."""
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.from_dict(raw)
