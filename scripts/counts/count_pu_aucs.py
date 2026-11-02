"""Validation ROC-AUC of every stage S2b configuration, with a substrate bootstrap (ADR 031).

The one read of the Chemlex **validation** labels for the PU models. Six configurations, three
feature blocks times two arms, against the baselines already read by
``scripts/counts/count_baseline_aucs.py``. The test side is never opened and the file name is
checked below.

**Why ROC-AUC and no tie-breaking.** It is the Mann-Whitney statistic: a tied pair contributes
exactly one half, so a heavily tied score column gives a correct figure whatever order the rows sit
in. That is not a detail here -- the ``morgan_centre`` configurations give 478 and 595 distinct
scores for 2,615 pairs. No enrichment at k is printed, because that one **does** depend on how a
tied group is ordered and an arbitrary order would be an uncontrolled parameter (ADR 031 item 4).

**Why the bootstrap resamples substrates and not pairs.** Validation pairs share acids and amines,
so their outcomes are correlated; resampling pairs would treat them as independent and give an
interval that is too narrow. Owen's pigeonhole bootstrap resamples the acids and the amines and
keeps the pairs both of whose substrates were drawn.

Usage:
    uv run python scripts/counts/count_pu_aucs.py ~/University/Phd/Dataset/Datasets \\
        ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if len(sys.argv) != 3:
    raise SystemExit(
        "usage: scripts/counts/count_pu_aucs.py <Datasets root> <validation parquet>"
    )
ROOT = Path(sys.argv[1]).expanduser()
VALID = Path(sys.argv[2]).expanduser()
SCORES = ROOT / "derived/private/scores/amide_coupling/HATU_DIPEA_DMF"
# Fixed here, before the run, so it is not chosen once the numbers are visible (ADR 031 item 5).
RESAMPLES = 2000
SEED = 20261102

# A file-name check and not a substring test on the path: the test side is never read, and `assert`
# would vanish under `python -O`.
if VALID.name != "validation.parquet":
    raise SystemExit(f"refusing {VALID.name}: only validation.parquet is ever read")


def pigeonhole_interval(
    labels: np.ndarray, scores: np.ndarray, acids: np.ndarray, amines: np.ndarray
) -> tuple[float, float]:
    """A 95 % interval from resampling the substrates, Owen's pigeonhole bootstrap.

    A resample draws acids and amines independently with replacement and keeps every pair whose two
    substrates were both drawn, so a popular substrate carries all of its pairs in or out together.
    A resample without both labels present cannot give an AUC and is skipped, with the count of
    skips printed, because silently dropping them would narrow the interval.
    """
    generator = np.random.default_rng(SEED)
    unique_acids, unique_amines = np.unique(acids), np.unique(amines)
    values, skipped = [], 0
    for _ in range(RESAMPLES):
        kept_a = set(generator.choice(unique_acids, len(unique_acids), replace=True))
        kept_b = set(generator.choice(unique_amines, len(unique_amines), replace=True))
        keep = np.array(
            [a in kept_a and b in kept_b for a, b in zip(acids, amines, strict=True)]
        )
        if keep.sum() < 2 or len(np.unique(labels[keep])) < 2:
            skipped += 1
            continue
        values.append(roc_auc_score(labels[keep], scores[keep]))
    if skipped:
        print(
            f"    ({skipped} of {RESAMPLES} resamples had only one class and were skipped)"
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    labels = pd.read_parquet(VALID, columns=["key_a_stereo", "key_b_stereo", "label"])
    print(
        f"validation pairs {len(labels):,}, positives {int(labels['label'].sum()):,} "
        f"({100 * labels['label'].mean():.1f} %); {RESAMPLES:,} substrate resamples, seed {SEED}"
    )
    print(
        f"\n{'configuration':30s} {'distinct':>9s} {'AUC':>7s} {'95 % interval':>18s}"
    )
    for path in sorted(SCORES.glob("pu_score_*.parquet")):
        scores = pd.read_parquet(
            path, columns=["score_set", "key_a_stereo", "key_b_stereo", "score"]
        )
        scores = scores[scores["score_set"] == "validation"]
        merged = labels.merge(
            scores, on=["key_a_stereo", "key_b_stereo"], validate="one_to_one"
        )
        if len(merged) != len(labels):
            raise SystemExit(
                f"{path.stem}: {len(merged)} of {len(labels)} validation pairs matched"
            )
        auc = roc_auc_score(merged["label"], merged["score"])
        low, high = pigeonhole_interval(
            merged["label"].to_numpy(),
            merged["score"].to_numpy(),
            merged["key_a_stereo"].to_numpy(),
            merged["key_b_stereo"].to_numpy(),
        )
        print(
            f"{path.stem:30s} {merged['score'].nunique():9,d} {auc:7.4f} "
            f"  [{low:.4f}, {high:.4f}]"
        )
    # The bars, from the journal's post-rebuild table of 2026-10-29 and reproducible with
    # `scripts/counts/count_baseline_aucs.py`. B8 is 0.7221 at the class width, NOT the 0.7115 that
    # was quoted in accepted records until the journal recorded that grep finds it in no source
    # file -- a number with no command behind it, which is exactly what this project keeps catching.
    print(
        "\nalready read, for comparison (scripts/counts/count_baseline_aucs.py, 2026-10-29): "
        "B6 0.8221, B4 0.7426, B8 0.7221 (class width), B2 0.632 (class width), B7 0.5982"
    )


if __name__ == "__main__":
    main()
