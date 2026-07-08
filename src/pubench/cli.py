"""Command-line entry point: run a PU benchmark from a YAML config."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pubench.runner import Experiment, load_config


def _configure_logging(level: str) -> None:
    """Configure structured console logging at ``level``."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the benchmark, and print the test-split comparison table.

    Parameters
    ----------
    argv:
        Optional argument vector (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description="Run a PU-learning benchmark (pubench).")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    config = load_config(args.config)
    table = Experiment(config).run()

    test_view = table.loc[table["split"] == "test"].sort_values("roc_auc", ascending=False)
    columns = [
        "model",
        "featurizer",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "neg_precision",
    ]
    print(test_view[columns].to_string(index=False))


if __name__ == "__main__":
    main()
