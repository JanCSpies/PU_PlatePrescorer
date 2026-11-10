"""Unlabeled space U for one class family, and the overlap report (ADR 003).

Reads the library (``<root>/private/library/<class>.parquet``), the positives
(``<root>/positives/<class>/positives.parquet``) and a benchmark validation file. Writes
``<root>/private/unlabeled/<class>/<family>.parquet`` and ``.report.json`` with how much the lab
substrates and the validation substrates overlap the positives (stereo-agnostic keys).

Usage:
    uv run python scripts/unlabeled.py amide_coupling HATU_DIPEA_DMF ROOT --validation VALIDATION.parquet
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa

from platepu.classify.engine import load_class
from platepu.io import write_report, write_table
from platepu.library.unlabeled import unlabeled_pairs


def share(keys: pd.Series, known: set[str]) -> dict[str, float | int]:
    """How many of the (unique) keys are in ``known``."""
    unique = set(keys)
    hits = len(unique & known)
    return {
        "n": len(unique),
        "in": hits,
        "share": round(hits / len(unique), 4) if unique else 0.0,
    }


def pair_groups(
    pairs: pd.DataFrame, acids: set[str], amines: set[str]
) -> dict[str, int]:
    """Pairs whose acid and amine are both / one / none among known substrates."""
    known = pairs["key_a_flat"].isin(acids).astype(int) + pairs["key_b_flat"].isin(
        amines
    ).astype(int)
    return {
        "both": int((known == 2).sum()),
        "one": int((known == 1).sum()),
        "none": int((known == 0).sum()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Build U and report the overlap of lab and validation substrates with the positives."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("family")
    parser.add_argument("root", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args(argv)

    rule = load_class(Path(f"configs/classes/{args.name}.yml"))
    library = pd.read_parquet(
        args.root / "private" / "library" / f"{args.name}.parquet"
    )
    positives = pd.read_parquet(
        args.root / "positives" / args.name / "positives.parquet"
    )
    family = positives[positives["family"] == args.family]
    everything = positives[positives["family"] == "all"]
    unlabeled = unlabeled_pairs(library, family, rule.role_a, rule.role_b)
    out = args.root / "private" / "unlabeled" / args.name / f"{args.family}.parquet"
    write_table(
        pa.Table.from_pandas(unlabeled, preserve_index=False), out, root=args.root
    )

    validation = pd.read_parquet(args.validation)
    lab_a = library.loc[library["role"] == rule.role_a, "key_flat"]
    lab_b = library.loc[library["role"] == rule.role_b, "key_flat"]
    report: dict[str, object] = {
        "stage": "unlabeled",
        "class": args.name,
        "family": args.family,
        "positives_pairs": len(family),
        "unlabeled_pairs": len(unlabeled),
        "lab_pairs_in_positives": int(len(lab_a) * len(lab_b) - len(unlabeled)),
    }
    for label, p in (("family", family), ("class_all_conditions", everything)):
        acids, amines = set(p["key_a_flat"]), set(p["key_b_flat"])
        report[f"lab_{rule.role_a}s_in_P_{label}"] = share(lab_a, acids)
        report[f"lab_{rule.role_b}s_in_P_{label}"] = share(lab_b, amines)
        report[f"validation_{rule.role_a}s_in_P_{label}"] = share(
            validation["key_a_flat"], acids
        )
        report[f"validation_{rule.role_b}s_in_P_{label}"] = share(
            validation["key_b_flat"], amines
        )
        report[f"validation_pairs_by_substrates_in_P_{label}"] = pair_groups(
            validation, acids, amines
        )
    report[f"lab_{rule.role_a}s_in_validation"] = share(
        lab_a, set(validation["key_a_flat"])
    )
    report[f"lab_{rule.role_b}s_in_validation"] = share(
        lab_b, set(validation["key_b_flat"])
    )
    write_report(out.with_suffix(".report.json"), report)
    print(
        f"{args.name} {args.family}: U {len(unlabeled):,} pairs, P {len(family):,} pairs"
    )


if __name__ == "__main__":
    main()
