"""Tests for scripts/score_pu.py, the stage S2a entry point, on tiny hand-written rows in a tmp root.

Written BEFORE the script exists (2026-11-02): until scripts/score_pu.py is written this file fails at
collection with ModuleNotFoundError, and that is intended. The chemistry (spies, threshold, flags) is
tested in tests/test_score_pu.py; here only the script's own work is pinned: which files it reads,
which columns, where it writes, what the report says, and what it refuses.

Every row is made up (open, hand-written; no Chemlex, Pistachio or inventory row). The builders are
this file's own, in the shape of tests/test_score_pu.py: "P-like" substrates share one fingerprint,
"odd" ones share another, so a pair of odd substrates looks like no positive. The config is a real
YAML written into ``tmp_path`` and read by ``load_pu_config``, because reading it is part of the job.

Fixture sizes: 8 positives of width ``all`` (0.25 of them is 2 spies) plus 4 positives of another
family that the script must not read, 8 unlabeled pairs, so the table holds 8 + 2 = 10 rows.
"""

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from score_pu import main as score_pu

from platepu.io import sha256
from platepu.score.pu_config import config_sha256, load_pu_config

CLASS = "amide_coupling"
STD = "std1"
OTHER_FAMILY = "HATU_DIPEA_DMF"
N_BITS = 8
P_LIKE_ACID = [1, 1, 0, 0, 0, 0, 0, 0]
P_LIKE_AMINE = [0, 0, 1, 1, 0, 0, 0, 0]
ODD_ACID = [0, 0, 0, 0, 1, 1, 0, 0]
ODD_AMINE = [0, 0, 0, 0, 0, 0, 1, 1]
TABLE_COLUMNS = [
    "key_a_stereo", "key_b_stereo", "key_a_flat", "key_b_flat",
    "in_p", "weight", "fold", "spy_score", "reliable_negative", "threshold",
    "model", "feature_block", "seed", "config_sha256", "std_version", "licence",
]  # fmt: skip
HEX64 = re.compile(r"[0-9a-f]{64}")

CONFIG_YAML = """\
model: pair_hgb
feature_block: {block}
p_width: all
unlabeled: u_stem
spy_fraction: 0.25
spy_recall: 0.95
spy_scoring: out_of_fold
n_folds: 2
learner: HistGradientBoostingClassifier
learner_params:
  max_iter: 20
  early_stopping: false
  min_samples_leaf: 1
seed: 7
"""


def pair_table(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """The four pair keys; the flat key is the stereo key in lower case."""
    return pd.DataFrame(
        {
            "key_a_stereo": [a for a, _ in pairs],
            "key_b_stereo": [b for _, b in pairs],
            "key_a_flat": [a.lower() for a, _ in pairs],
            "key_b_flat": [b.lower() for _, b in pairs],
        }
    )


def substrate_table() -> pa.Table:
    """Eight P-like and four odd substrates, fingerprints as fixed_size_list<uint8>[8] as stage F."""
    rows = (
        [("acid", f"PA{i}", P_LIKE_ACID, 20) for i in range(4)]
        + [("amine", f"PN{i}", P_LIKE_AMINE, 10) for i in range(4)]
        + [("acid", f"OA{i}", ODD_ACID, 10) for i in range(2)]
        + [("amine", f"ON{i}", ODD_AMINE, 5) for i in range(2)]
    )
    bits = np.array([row[2] for row in rows], dtype=np.uint8)
    frame = pd.DataFrame(
        {
            "key_stereo": [row[1] for row in rows],
            "role": [row[0] for row in rows],
            "n_heavy_atoms": np.array([row[3] for row in rows], dtype="int16"),
            "std_version": STD,
        }
    )
    centre = pa.FixedSizeListArray.from_arrays(pa.array(bits.ravel()), N_BITS)
    plain = pa.FixedSizeListArray.from_arrays(pa.array(bits.ravel() * 2), N_BITS)
    near = pa.FixedSizeListArray.from_arrays(pa.array(bits.ravel() * 3), N_BITS)
    return (
        pa.Table.from_pandas(frame, preserve_index=False)
        .append_column("morgan_centre", centre)
        .append_column("morgan", plain)
        .append_column("morgan_near", near)
    )


def write_inputs(
    root: Path,
    *,
    positives_licence: str = "open",
    positives_std: str = STD,
    unlabeled_std: str = STD,
    substrates_std: str = STD,
    unlabeled_stem: str = "u_stem",
    with_substrates: bool = True,
) -> None:
    """Write the three inputs under ``root`` at the designed paths; no benchmark file anywhere."""
    p_like = list(
        itertools.product([f"PA{i}" for i in range(4)], [f"PN{i}" for i in range(4)])
    )
    odd = list(itertools.product(["OA0", "OA1"], ["ON0", "ON1"]))
    wanted = pair_table(p_like[:8]).assign(family="all", n_reports=1)
    unwanted = pair_table(p_like[8:12]).assign(family=OTHER_FAMILY, n_reports=1)
    positives = pd.concat([wanted, unwanted], ignore_index=True).assign(
        std_version=positives_std, licence="open"
    )
    positives.loc[0, "licence"] = positives_licence
    unlabeled = pair_table(p_like[12:16] + odd).assign(
        n_draws=1, std_version=unlabeled_std, licence="open"
    )
    positives_dir = root / "positives" / CLASS
    unlabeled_dir = root / "unlabeled" / CLASS
    features_dir = root / "private" / "features" / CLASS
    for folder in (positives_dir, unlabeled_dir, features_dir):
        folder.mkdir(parents=True)
    positives.to_parquet(positives_dir / "positives.parquet")
    unlabeled.to_parquet(unlabeled_dir / f"{unlabeled_stem}.parquet")
    if with_substrates:
        table = substrate_table()
        table = table.set_column(
            table.schema.get_field_index("std_version"),
            "std_version",
            pa.array([substrates_std] * table.num_rows),
        )
        pq.write_table(table, features_dir / "substrates.parquet")


def write_config(
    directory: Path, name: str = "pu_test.yml", block: str = "morgan_centre"
) -> Path:
    """A real stage S2 YAML in ``directory``."""
    path = directory / name
    path.write_text(CONFIG_YAML.format(block=block), encoding="utf-8")
    return path


def run(root: Path, config: Path) -> Path:
    """Run the script; return the folder it writes to."""
    score_pu([CLASS, str(root), "--config", str(config)])
    return root / "pu" / CLASS


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_negatives_land_at_the_designed_path_with_the_designed_columns(
    tmp_path: Path,
) -> None:
    """The table is pu/<class>/<config stem>.negatives.parquet outside private/ (every row is open),
    with the S2a columns in order, one row per U pair plus one per spy (8 + 2), and a valid report
    beside it."""
    root = tmp_path / "derived"
    write_inputs(root)
    config = write_config(tmp_path)
    out = run(root, config)
    table = pq.read_table(out / "pu_test.negatives.parquet")
    assert table.column_names == TABLE_COLUMNS
    frame = table.to_pandas()
    assert len(frame) == 10
    assert int(frame["in_p"].sum()) == 2
    assert set(frame["licence"]) == {"open"}
    assert not (root / "private" / "pu").exists()
    assert isinstance(read_report(out / "pu_test.negatives.report.json"), dict)


def test_the_unlabeled_rows_are_the_table_and_the_other_family_is_not_read(
    tmp_path: Path,
) -> None:
    """Every U pair appears once, not in P. The 4 positives of another family are not P: read as P
    there would be 12 positives, 3 spies and 11 rows. The 2 spies come from the 8 of width 'all'."""
    root = tmp_path / "derived"
    write_inputs(root)
    out = run(root, write_config(tmp_path))
    frame = pd.read_parquet(out / "pu_test.negatives.parquet")
    unlabeled = pd.read_parquet(root / "unlabeled" / CLASS / "u_stem.parquet")
    positives = pd.read_parquet(root / "positives" / CLASS / "positives.parquet")
    keys = ["key_a_stereo", "key_b_stereo"]
    in_u = frame.loc[~frame["in_p"], keys].apply(tuple, axis=1)
    assert sorted(in_u) == sorted(unlabeled[keys].apply(tuple, axis=1))
    spies = frame.loc[frame["in_p"], keys].apply(tuple, axis=1)
    wanted = positives.loc[positives["family"] == "all", keys].apply(tuple, axis=1)
    assert set(spies) <= set(wanted)


def test_it_reads_no_benchmark_at_all(tmp_path: Path) -> None:
    """S2a cannot read a label even by accident (ADR 008 item 6): the tree holds no validation.parquet,
    labels.parquet or test.parquet anywhere and the run succeeds; and the parser has no benchmark
    argument, so handing it one is refused (argparse exits with status 2)."""
    root = tmp_path / "derived"
    write_inputs(root)
    config = write_config(tmp_path)
    assert not [
        path
        for path in root.rglob("*.parquet")
        if path.name in {"validation.parquet", "labels.parquet", "test.parquet"}
    ]
    run(root, config)
    with pytest.raises(SystemExit) as refused:
        score_pu([CLASS, str(root), "--config", str(config), "--validation", "x"])
    assert refused.value.code == 2


def test_config_is_required(tmp_path: Path) -> None:
    """No --config, no run: the choices are in the file and nowhere else (CLAUDE.md §6)."""
    root = tmp_path / "derived"
    write_inputs(root)
    with pytest.raises(SystemExit) as refused:
        score_pu([CLASS, str(root)])
    assert refused.value.code == 2


def test_a_licensed_positive_is_refused_outside_private(tmp_path: Path) -> None:
    """ADR 009 guard, as io.write_table actually behaves: one 'licensed' positives row makes the
    whole table 'licensed' (strictest of P and U), and the script writes under pu/, not under
    private/, so write_table raises ValueError and neither table nor report is written. (The script
    does not move the output under private/ on its own; if it ever does, this test must change with
    that decision, not be loosened.)"""
    root = tmp_path / "derived"
    write_inputs(root, positives_licence="licensed")
    with pytest.raises(ValueError, match=r"private/"):
        run(root, write_config(tmp_path))
    assert not list((root / "pu").rglob("*"))


def test_two_runs_give_equal_tables_and_reports(tmp_path: Path) -> None:
    """Same inputs, same config, same seed, two roots: the tables are identical and the reports
    agree except for written_at."""
    config = write_config(tmp_path)
    outs = []
    for name in ("first", "second"):
        root = tmp_path / name
        write_inputs(root)
        outs.append(run(root, config))
    first, second = (pd.read_parquet(out / "pu_test.negatives.parquet") for out in outs)
    pd.testing.assert_frame_equal(first, second)
    reports = [read_report(out / "pu_test.negatives.report.json") for out in outs]
    for report in reports:
        report.pop("written_at")
    assert reports[0] == reports[1]


def test_both_config_hashes_are_recorded_and_differ(tmp_path: Path) -> None:
    """Luka's decision of 2026-11-02: the table's config_sha256 hashes the config's VALUES
    (pu_config.config_sha256); write_report hashes the config FILE. Both are 64 hex characters and
    they are different strings, so the record carries both readings and none is mistaken for the
    other."""
    root = tmp_path / "derived"
    write_inputs(root)
    config = write_config(tmp_path)
    out = run(root, config)
    frame = pd.read_parquet(out / "pu_test.negatives.parquet")
    report = read_report(out / "pu_test.negatives.report.json")
    (of_values,) = set(frame["config_sha256"])
    assert of_values == config_sha256(load_pu_config(config))
    of_file = report["configs"][str(config)]
    assert of_file == sha256(config)
    assert HEX64.fullmatch(of_values)
    assert HEX64.fullmatch(of_file)
    assert of_values != of_file
    assert of_values != hashlib.sha256(config.read_bytes()).hexdigest()


def test_the_config_stem_names_the_output(tmp_path: Path) -> None:
    """Two configs that differ only in feature_block write two tables, named by their stems, and the
    second run leaves the first table byte for byte as it was."""
    root = tmp_path / "derived"
    write_inputs(root)
    centre = write_config(tmp_path, "pu_centre.yml", "morgan_centre")
    plain = write_config(tmp_path, "pu_plain.yml", "morgan")
    out = run(root, centre)
    first_path = out / "pu_centre.negatives.parquet"
    first_bytes = first_path.read_bytes()
    run(root, plain)
    assert first_path.read_bytes() == first_bytes
    second = pd.read_parquet(out / "pu_plain.negatives.parquet")
    assert set(second["feature_block"]) == {"morgan"}
    assert set(pd.read_parquet(first_path)["feature_block"]) == {"morgan_centre"}
    run(root, write_config(tmp_path, "pu_near.yml", "morgan_near"))
    near = pd.read_parquet(out / "pu_near.negatives.parquet")
    assert set(near["feature_block"]) == {"morgan_near"}
    assert (out / "pu_centre.negatives.report.json").exists()
    assert (out / "pu_plain.negatives.report.json").exists()


def test_a_u_stem_that_names_no_table_raises_with_the_path(tmp_path: Path) -> None:
    """The config names u_stem, only another_stem.parquet exists: the error carries the path it
    looked for, so nobody has to guess which of three inputs is missing."""
    root = tmp_path / "derived"
    write_inputs(root, unlabeled_stem="another_stem")
    missing = root / "unlabeled" / CLASS / "u_stem.parquet"
    with pytest.raises((FileNotFoundError, ValueError), match=re.escape(str(missing))):
        run(root, write_config(tmp_path))


def test_a_missing_feature_table_raises_with_the_path(tmp_path: Path) -> None:
    """No private/features/<class>/substrates.parquet: the error names that path."""
    root = tmp_path / "derived"
    write_inputs(root, with_substrates=False)
    missing = root / "private" / "features" / CLASS / "substrates.parquet"
    with pytest.raises((FileNotFoundError, ValueError), match=re.escape(str(missing))):
        run(root, write_config(tmp_path))


@pytest.mark.parametrize("where", ["positives", "unlabeled", "substrates"])
def test_a_mixed_std_version_raises(tmp_path: Path, where: str) -> None:
    """One standardisation for P, U and the features (CLAUDE.md §6): any one of the three on std2
    stops the run with pair_matrix's error naming std_version, and nothing is written."""
    root = tmp_path / "derived"
    write_inputs(root, **{f"{where}_std": "std2"})
    with pytest.raises(ValueError, match="std_version"):
        run(root, write_config(tmp_path))
    assert not list((root / "pu").rglob("*.parquet"))


def test_report_counts_match_the_table(tmp_path: Path) -> None:
    """rows_out is the table's length (10); rows_in holds each input's row count as the script reads
    it: 8 positives of width 'all' (not the 12 in the file), 8 unlabeled, 12 substrates;
    n_reliable_negatives is the table's reliable_negative sum; the report also names the class, the
    config path and the std_version."""
    root = tmp_path / "derived"
    write_inputs(root)
    config = write_config(tmp_path)
    out = run(root, config)
    frame = pd.read_parquet(out / "pu_test.negatives.parquet")
    report = read_report(out / "pu_test.negatives.report.json")
    assert report["rows_out"] == len(frame) == 10
    assert report["rows_in"] == {"positives": 8, "unlabeled": 8, "substrates": 12}
    assert report["n_reliable_negatives"] == int(frame["reliable_negative"].sum())
    assert report["class"] == CLASS
    assert report["config"] == str(config)
    assert report["std_version"] == STD
    assert "stage" in report
