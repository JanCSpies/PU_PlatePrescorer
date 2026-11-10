"""Test for scripts/features.py on tiny hand-written inputs in a tmp root (open, made-up keys)."""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from features import main as features

STD = "std1"


def collect_keys(value: object) -> set[str]:
    """All dictionary keys in a nested JSON report."""
    if not isinstance(value, dict):
        return set()
    return set(value).union(*(collect_keys(item) for item in value.values()))


def write_inputs(tmp_path: Path, validation_name: str) -> Path:
    """Write the tiny positives, library and validation tables under tmp_path; return the validation path."""
    tautomer_key = "BXHCJLRTXPHUGH-UHFFFAOYSA-N"
    pair = {
        "key_a_flat": "x",
        "key_b_flat": "y",
        "licence": "open",
        "std_version": STD,
    }
    positives = pd.DataFrame(
        [
            {**pair, "family": "all", "key_a_stereo": "A1", "substrate_a": "CC(=O)O",
             "key_b_stereo": "N1", "substrate_b": "NCc1ccccc1"},
            {**pair, "family": "all", "key_a_stereo": tautomer_key, "substrate_a": "O=C(O)c1ccnc(O)c1",
             "key_b_stereo": "N1", "substrate_b": "NCc1ccccc1"},
            {**pair, "family": "all", "key_a_stereo": tautomer_key, "substrate_a": "O=C(O)c1cc[nH]c(=O)c1",
             "key_b_stereo": "N1", "substrate_b": "NCc1ccccc1"},
        ]
    )  # fmt: skip
    library = pd.DataFrame(
        [
            {"key_stereo": "A1", "key_flat": "a1", "smiles": "CC(=O)O", "role": "acid",
             "licence": "internal", "std_version": STD},
            {"key_stereo": "N2", "key_flat": "n2", "smiles": "C1COCCN1", "role": "amine",
             "licence": "internal", "std_version": STD},
        ]
    )  # fmt: skip
    # ADR 009: the benchmark writes a licence column; Chemlex is noncommercial. The row is the
    # positive pair A1 + N1, so both substrates are also in positives (open).
    validation = (
        positives.drop(columns=["family"]).iloc[:1].assign(licence="noncommercial")
    )

    (tmp_path / "positives" / "amide_coupling").mkdir(parents=True)
    (tmp_path / "private" / "library").mkdir(parents=True)
    positives.to_parquet(
        tmp_path / "positives" / "amide_coupling" / "positives.parquet"
    )
    library.to_parquet(tmp_path / "private" / "library" / "amide_coupling.parquet")
    validation_path = tmp_path / validation_name
    validation.to_parquet(validation_path)

    return validation_path


def test_features_written_under_private_with_fixed_size_morgan(tmp_path: Path) -> None:
    """Three acids (one tautomer pair under one key), two amines: 4 rows out, 1 key with 2 SMILES;
    the table is under private/ because the library is internal, Morgan is fixed_size_list<uint8>[2048]."""
    validation_path = write_inputs(tmp_path, "validation.parquet")
    features(["amide_coupling", str(tmp_path), "--validation", str(validation_path)])

    out = tmp_path / "private" / "features" / "amide_coupling"
    table = pq.read_table(out / "substrates.parquet")
    assert table.num_rows == 4  # acids A1, tautomer key; amines N1, N2
    assert table.schema.field("morgan").type == pa.list_(pa.uint8(), 2048)
    assert table.schema.field("morgan_centre").type == pa.list_(pa.uint8(), 2048)
    report = json.loads((out / "substrates.report.json").read_text(encoding="utf-8"))
    assert {"rows_out", "keys_with_several_smiles"} <= collect_keys(report)
    assert not (tmp_path / "features").exists()


def test_substrate_measured_by_chemlex_is_noncommercial_in_the_features(
    tmp_path: Path,
) -> None:
    """N1 is an open positive substrate and is in the noncommercial validation row, so it comes out
    noncommercial; N2 is only in the (internal) library; A1 is in the library too, so internal."""
    validation_path = write_inputs(tmp_path, "validation.parquet")
    features(["amide_coupling", str(tmp_path), "--validation", str(validation_path)])
    table = pq.read_table(
        tmp_path / "private" / "features" / "amide_coupling" / "substrates.parquet"
    ).to_pandas()
    licence = table.set_index("key_stereo")["licence"]
    assert (licence["N1"], licence["N2"], licence["A1"]) == (
        "noncommercial",
        "internal",
        "internal",
    )


def test_features_refuses_a_test_file(tmp_path: Path) -> None:
    """A --validation file called test.parquet (the Chemlex test side) is refused before any read."""
    test_path = write_inputs(tmp_path, "test.parquet")
    with pytest.raises(ValueError, match=r"validation\.parquet"):
        features(["amide_coupling", str(tmp_path), "--validation", str(test_path)])
    assert not (tmp_path / "private" / "features").exists()
