"""Summarise all transfer experiment artifacts into one comparison table.

Reads every ``artifacts/transfer_*.json`` (written by scripts/06) and prints a combined
config × arm table of VAL/TEST ROC-AUC and AUPRC, highlighting the two headline comparisons:

* **spy (PAYN-parity, SCAR) vs pu_coral_iw (the domain-adapted "better PAYN")** — does adaptation
  beat the in-domain baseline?
* **lit-only vs lit+ORD** — does the mined C-N data expansion help transfer?

Model selection is on VAL; TEST is shown for the final honest report only. Usage:
    uv run python scripts/09_summarize_transfer.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACTS_DIR = Path("artifacts")


def _load_configs() -> list[dict[str, Any]]:
    """Load every transfer artifact, tagged by its filename stem."""
    configs = []
    for path in sorted(ARTIFACTS_DIR.glob("transfer_*.json")):
        payload = json.loads(path.read_text())
        payload["_config"] = path.stem.replace("transfer_", "")
        configs.append(payload)
    return configs


def main() -> None:
    """Entry point: print the combined transfer comparison table."""
    configs = _load_configs()
    if not configs:
        print("No artifacts/transfer_*.json found — run scripts/06 first.")
        return

    for cfg in configs:
        summary = cfg["summary"]
        auc = summary.get("domain_auc_source_vs_grid", float("nan"))
        n_pos = summary.get("n_positives_total", "?")
        print(f"\n=== {cfg['_config']}  (positives={n_pos}, domain-shortcut AUC={auc:.3f}) ===")
        print(f"{'arm':<14} {'family':<11} {'VAL_AUC':>7} {'VAL_AP':>7} {'TE_AUC':>7} {'TE_AP':>7}")
        print("-" * 60)
        arms = cfg["arms"]
        best = max(arms, key=lambda a: a["val"]["auprc"])
        for arm in sorted(arms, key=lambda a: a["val"]["auprc"], reverse=True):
            test = arm["test"] or {}
            mark = "  <- best(VAL)" if arm["name"] == best["name"] else ""
            te_auc = test.get("roc_auc", float("nan"))
            te_ap = test.get("auprc", float("nan"))
            print(
                f"{arm['name']:<14} {arm['family']:<11} "
                f"{arm['val']['roc_auc']:>7.3f} {arm['val']['auprc']:>7.3f} "
                f"{te_auc:>7.3f} {te_ap:>7.3f}{mark}"
            )

    # Headline comparisons across configs.
    print("\n=== headline comparisons (TEST ROC-AUC) ===")
    for cfg in configs:
        arm_map: dict[str, Any] = {a["name"]: a for a in cfg["arms"]}

        def test_auc(name: str, table: dict[str, Any]) -> float:
            arm = table.get(name) or {}
            test = arm.get("test") or {}
            return float(test.get("roc_auc", float("nan")))

        print(
            f"  {cfg['_config']:<20} "
            f"spy(PAYN)={test_auc('pu_spy', arm_map):.3f}  "
            f"pu_iw(better-PAYN)={test_auc('pu_iw', arm_map):.3f}  "
            f"sup_coral={test_auc('sup_coral', arm_map):.3f}"
        )


if __name__ == "__main__":
    main()
