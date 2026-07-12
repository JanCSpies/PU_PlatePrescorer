"""Pool every Chemlex PU-method result with the HistGB champion into one leaderboard.

Reads the per-method blocks written by :mod:`scripts_JSP.run_chemlex_all_models`
(``artifacts/dataset_switch/allmodels/*.json``) plus the supervised champion already stored in
``multiseed_metrics.json``, and prints / persists a head-to-head table of TEST ROC-AUC (mean +- std
over 5 seeds) for every regime x featurizer cell, sorted by the production cell (raw_ecfp). The
point of the table: on an in-scope target (Chemlex amide), does any PU learner beat the naive
P-vs-U HistGB baseline? The ``delta`` column is each method minus the champion in that cell.

Usage:
    uv run python -m scripts_JSP.summarize_chemlex_models
"""

from __future__ import annotations

import json
from pathlib import Path

ALLMODELS = Path("artifacts/dataset_switch/allmodels")
CHAMPION = Path("artifacts/dataset_switch/multiseed_metrics.json")
OUT_MD = Path("artifacts/dataset_switch/chemlex_model_comparison.md")

REGIMES = ("transductive", "inductive")
FEATURIZERS = ("raw_ecfp", "corr08")
#: Cell used to rank the leaderboard (the production featurizer / plate-pre-scoring regime).
RANK_REGIME, RANK_FEAT = "transductive", "raw_ecfp"


def _cell(block: dict[str, object]) -> tuple[float, float]:
    """Return ``(mean_auc, std_auc)`` from a per-method metric block."""
    mean = block["mean"]  # type: ignore[index]
    std = block["std"]  # type: ignore[index]
    return float(mean["roc_auc"]), float(std["roc_auc"])  # type: ignore[index]


def _load_models() -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Load champion + every PU method as ``label -> metrics[regime][featurizer] -> block``."""
    models: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    champ = json.loads(CHAMPION.read_text())["metrics"]
    models["sup_histgb_l2 (champion)"] = champ
    for path in sorted(ALLMODELS.glob("*.json")):
        payload = json.loads(path.read_text())
        label = payload["model_label"]
        if label == "sup_histgb_l2":  # an in-batch control, if it was run; keep it distinct
            label = "sup_histgb_l2 (rerun)"
        models[label] = payload["metrics"]
    return models


def main() -> None:
    """Assemble and emit the leaderboard (stdout + markdown)."""
    models = _load_models()
    champ_metrics = models["sup_histgb_l2 (champion)"]
    champ_auc = {(r, f): _cell(champ_metrics[r][f])[0] for r in REGIMES for f in FEATURIZERS}

    # Rank all methods by the production cell.
    ranked = sorted(
        models.items(),
        key=lambda kv: _cell(kv[1][RANK_REGIME][RANK_FEAT])[0],
        reverse=True,
    )

    header = f"{'model':<26}"
    for r in REGIMES:
        for f in FEATURIZERS:
            header += f" {r[:5]}/{f[:7]:<11}"
    header += f" {'Δ vs champ':<11}"
    lines = [header, "-" * len(header)]
    for label, metrics in ranked:
        row = f"{label:<26}"
        for r in REGIMES:
            for f in FEATURIZERS:
                m, s = _cell(metrics[r][f])
                row += f" {m:.3f}±{s:.3f} "
        delta = _cell(metrics[RANK_REGIME][RANK_FEAT])[0] - champ_auc[(RANK_REGIME, RANK_FEAT)]
        row += f" {delta:+.3f}"
        lines.append(row)
    table = "\n".join(lines)
    print(table)

    # Markdown for the journal.
    md = [
        "# Chemlex amide: all PU methods vs the HistGB champion (TEST ROC-AUC, 5 seeds)",
        "",
        f"Ranked by {RANK_REGIME}/{RANK_FEAT}. Delta = method - champion in that cell.",
        "",
        "| model | "
        + " | ".join(f"{r}/{f}" for r in REGIMES for f in FEATURIZERS)
        + " | Δ vs champ |",
        "|" + "---|" * (len(REGIMES) * len(FEATURIZERS) + 2),
    ]
    for label, metrics in ranked:
        cells = " | ".join(
            f"{_cell(metrics[r][f])[0]:.3f}±{_cell(metrics[r][f])[1]:.3f}"
            for r in REGIMES
            for f in FEATURIZERS
        )
        delta = _cell(metrics[RANK_REGIME][RANK_FEAT])[0] - champ_auc[(RANK_REGIME, RANK_FEAT)]
        md.append(f"| {label} | {cells} | {delta:+.3f} |")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
