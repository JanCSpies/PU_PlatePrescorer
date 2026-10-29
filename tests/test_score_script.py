"""Test for scripts/score_baselines.py on tiny hand-written inputs in a tmp root.

Keys are made up; fingerprints are bit sets as in test_score_baselines.py. Nothing comes from the
library, the inventory or Chemlex. The script reads the real configs/models/baselines.yml, the way
tests/test_features_script.py reads the real class YAML, so these tests run from the repository root.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import score_baselines as score_module
from score_baselines import PAIR_COLUMNS, VALIDATION_COLUMNS
from score_baselines import main as score_baselines

STD = "std1"
FAMILY = "HATU_DIPEA_DMF"
N_BITS = 2048
BITS = {
    "A1": [0, 1, 2, 3],
    "A2": [0, 1, 4, 5],
    "A3": [10, 11, 12, 13],
    "N1": [100, 101, 102, 103],
    "N2": [100, 101, 104, 105],
    "N3": [200, 201, 202, 203],
}
SUBCLASS = {
    "A1": "aryl", "A2": "alpha_vinyl", "A3": "aryl",
    "N1": "aniline", "N2": "aliphatic_primary", "N3": "aniline",
}  # fmt: skip
SIZE = {"A1": 8, "A2": 9, "A3": 12, "N1": 7, "N2": 6, "N3": 10}
MODELS = {"B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"}
NO_BLOCK = {"B0", "B1", "B4", "B5", "B6", "B7"}


def collect_keys(value: object) -> set[str]:
    """All dictionary keys in a nested JSON report."""
    if isinstance(value, list):
        return set().union(*(collect_keys(item) for item in value))
    if not isinstance(value, dict):
        return set()
    return set(value).union(*(collect_keys(item) for item in value.values()))


def feature_table() -> pa.Table:
    """The six substrates as stage F writes them, Morgan as fixed_size_list<uint8>[2048]."""
    keys = list(BITS)
    counts = np.zeros((len(keys), N_BITS), dtype=np.uint8)
    for row, key in enumerate(keys):
        counts[row, BITS[key]] = 1
    table = pd.DataFrame(
        {
            "key_stereo": keys,
            "key_flat": [key.lower() for key in keys],
            "role": ["acid" if key[0] == "A" else "amine" for key in keys],
            "subclass": [SUBCLASS[key] for key in keys],
            "n_heavy_atoms": np.array([SIZE[key] for key in keys], dtype="int16"),
            "n_heavy_at_2_bonds": np.array([i % 3 for i in range(6)], dtype="int16"),
            "n_ewg_on_ring": np.array([i % 2 for i in range(6)], dtype="int16"),
            "licence": "internal",
            "std_version": STD,
        }
    )
    morgan = pa.FixedSizeListArray.from_arrays(
        pa.array(counts.ravel(), pa.uint8()), N_BITS
    )
    centre = counts.copy()
    centre[1, :4] = [
        1,
        1,
        1,
        0,
    ]  # A2 rooted differently from A2 on morgan, so B8 can differ from B2
    centre_array = pa.FixedSizeListArray.from_arrays(
        pa.array(centre.ravel(), pa.uint8()), N_BITS
    )
    return (
        pa.Table.from_pandas(table, preserve_index=False)
        .append_column("morgan", morgan)
        .append_column("morgan_centre", centre_array)
    )


def side_keys(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """The four key columns of a list of pairs."""
    return pd.DataFrame(
        {
            "key_a_stereo": [a for a, _ in pairs],
            "key_b_stereo": [b for _, b in pairs],
            "key_a_flat": [a.lower() for a, _ in pairs],
            "key_b_flat": [b.lower() for _, b in pairs],
        }
    )


def write_inputs(
    tmp_path: Path, validation_name: str, licence: str = "noncommercial"
) -> Path:
    """Write features, positives (class-wide and HATU), U and validation; return the validation path."""
    positives = pd.concat(
        [
            side_keys([("A1", "N1"), ("A2", "N2")]).assign(
                family="all", n_reports=[3, 1]
            ),
            side_keys([("A1", "N1")]).assign(family=FAMILY, n_reports=2),
        ],
        ignore_index=True,
    ).assign(licence="open", std_version=STD)
    unlabeled = side_keys([("A1", "N2"), ("A2", "N1"), ("A3", "N3")]).assign(
        substrate_a="smiles-a", substrate_b="smiles-b", licence="internal"
    )  # the real inventory table also carries these three, and no std_version
    validation = side_keys([("A1", "N1"), ("A2", "N3")]).assign(
        std_version=STD, label=[1, 0], conversion_mean=[55.0, 3.0], licence=licence
    )

    features_dir = tmp_path / "private" / "features" / "amide_coupling"
    unlabeled_dir = tmp_path / "private" / "unlabeled" / "amide_coupling"
    (tmp_path / "positives" / "amide_coupling").mkdir(parents=True)
    features_dir.mkdir(parents=True)
    unlabeled_dir.mkdir(parents=True)
    positives.to_parquet(
        tmp_path / "positives" / "amide_coupling" / "positives.parquet"
    )
    pq.write_table(feature_table(), features_dir / "substrates.parquet")
    unlabeled.to_parquet(unlabeled_dir / f"{FAMILY}.parquet")
    validation_path = tmp_path / validation_name
    validation.to_parquet(validation_path)
    return validation_path


def run(tmp_path: Path, validation_path: Path) -> Path:
    """Run the script; return the folder it writes to."""
    score_baselines(
        ["amide_coupling", FAMILY, str(tmp_path), "--validation", str(validation_path)]
    )
    return tmp_path / "private" / "scores" / "amide_coupling" / FAMILY


def test_score_script_refuses_a_test_file(tmp_path: Path) -> None:
    """A --validation file called test.parquet (the Chemlex test side) is refused before any read,
    and nothing is written."""
    test_path = write_inputs(tmp_path, "test.parquet")
    with pytest.raises(ValueError, match=r"validation\.parquet"):
        run(tmp_path, test_path)
    assert not (tmp_path / "private" / "scores").exists()


def test_scores_written_under_private(tmp_path: Path) -> None:
    """One table baselines.parquet and its report under private/scores/<class>/<family>/, with the
    agreed columns, float32 score, 2 validation pairs and 3 inventory pairs per model-and-width,
    and nothing under a scores/ folder outside private/. Inventory rows carry licence internal."""
    out = run(tmp_path, write_inputs(tmp_path, "validation.parquet"))
    table = pq.read_table(out / "baselines.parquet")
    assert table.column_names == [
        "score_set", "key_a_stereo", "key_b_stereo", "score", "model", "feature_block",
        "p_width", "seed", "config_sha256", "std_version", "licence",
    ]  # fmt: skip
    assert table.schema.field("score").type == pa.float32()
    frame = table.to_pandas()
    assert set(frame["model"]) == MODELS
    assert set(frame["score_set"]) == {"validation", "inventory"}
    one = frame[(frame["model"] == "B7") & (frame["score_set"] == "validation")]
    assert len(one) == 2
    assert set(frame.loc[frame["score_set"] == "inventory", "licence"]) == {"internal"}
    assert (out / "baselines.report.json").exists()
    assert not (tmp_path / "scores").exists()


def test_validation_scores_carry_the_licence_of_the_validation_file(
    tmp_path: Path,
) -> None:
    """ADR 009 item 5: the validation table says noncommercial (Chemlex), so its score rows do too;
    the script no longer stamps them open. Inventory rows stay internal."""
    out = run(tmp_path, write_inputs(tmp_path, "validation.parquet"))
    frame = pq.read_table(out / "baselines.parquet").to_pandas()
    by_set = frame.groupby("score_set")["licence"].agg(set)
    assert by_set["validation"] == {"noncommercial"}
    assert by_set["inventory"] == {"internal"}


def test_validation_scores_follow_whatever_licence_the_file_holds(
    tmp_path: Path,
) -> None:
    """Read, not hard-coded: a validation file labelled internal gives internal score rows."""
    out = run(
        tmp_path, write_inputs(tmp_path, "validation.parquet", licence="internal")
    )
    frame = pq.read_table(out / "baselines.parquet").to_pandas()
    assert set(frame.loc[frame["score_set"] == "validation", "licence"]) == {"internal"}


def test_both_scored_sets_are_read_with_their_licence() -> None:
    """ADR 009 item 5, stop stamping and start reading: both explicit column lists carry licence, so
    the script writes no licence value of its own (the inventory table is read as internal)."""
    assert "licence" in PAIR_COLUMNS
    assert "licence" in VALIDATION_COLUMNS


def test_b2_b3_and_b8_appear_at_both_p_widths_and_the_others_once(
    tmp_path: Path,
) -> None:
    """B2, B3 and B8: p_width is the family and 'all', so two groups each; B0, B1, B4, B5, B6, B7 read
    no P pair-table width, so one group with a null p_width. 9 models, 12 groups, 5 pairs each."""
    out = run(tmp_path, write_inputs(tmp_path, "validation.parquet"))
    frame = pd.read_parquet(out / "baselines.parquet")
    for model in ("B2", "B3", "B8"):
        assert set(frame.loc[frame["model"] == model, "p_width"]) == {FAMILY, "all"}
    for model in NO_BLOCK:
        assert frame.loc[frame["model"] == model, "p_width"].isna().all()
    assert len(frame) == 12 * 5
    assert (frame.groupby(["model", "p_width"], dropna=False).size() == 5).all()


def test_b8_runs_at_both_p_widths(tmp_path: Path) -> None:
    """B8 is B2's definition on another block, so like B2 it is scored at the family and at 'all',
    for both sets: 5 pairs at each width."""
    out = run(tmp_path, write_inputs(tmp_path, "validation.parquet"))
    frame = pd.read_parquet(out / "baselines.parquet")
    b8 = frame[frame["model"] == "B8"]
    assert set(b8["p_width"]) == {FAMILY, "all"}
    assert (b8.groupby("p_width").size() == 5).all()


def test_b8_differs_from_b2_in_the_table(tmp_path: Path) -> None:
    """The script reads the morgan_centre column for B8: inventory pair (A1,N2) at width 'all'
    (P = (A1,N1), (A2,N2)) is 1/3 for B2 and 3/5 for B8, because A2 and N2 are rooted differently."""
    out = run(tmp_path, write_inputs(tmp_path, "validation.parquet"))
    frame = pd.read_parquet(out / "baselines.parquet")
    rows = frame[(frame["p_width"] == "all") & (frame["key_a_stereo"] == "A1")]
    rows = rows[rows["model"].isin(["B2", "B8"]) & (rows["score_set"] == "inventory")]
    by_model = rows.set_index("model")["score"]
    assert by_model["B2"] != by_model["B8"]


def test_feature_block_is_set_for_the_fingerprint_baselines(tmp_path: Path) -> None:
    """B2 and B3 read morgan, B8 reads morgan_centre, at both widths and in both sets."""
    out = run(tmp_path, write_inputs(tmp_path, "validation.parquet"))
    frame = pd.read_parquet(out / "baselines.parquet")
    blocks = frame.groupby("model")["feature_block"].agg(set)
    assert blocks["B2"] == {"morgan"}
    assert blocks["B3"] == {"morgan"}
    assert blocks["B8"] == {"morgan_centre"}


def test_feature_block_is_null_for_the_rest(tmp_path: Path) -> None:
    """B0, B1, B4, B5, B6 and B7 read no fingerprint, so feature_block is null: not 'none', not an
    empty string, nothing that reads as a block name."""
    out = run(tmp_path, write_inputs(tmp_path, "validation.parquet"))
    frame = pd.read_parquet(out / "baselines.parquet")
    for model in NO_BLOCK:
        assert frame.loc[frame["model"] == model, "feature_block"].isna().all()


def test_score_table_columns_are_the_documented_order(tmp_path: Path) -> None:
    """docs/design/modeling.md "Output": feature_block sits between model and p_width."""
    out = run(tmp_path, write_inputs(tmp_path, "validation.parquet"))
    assert pq.read_table(out / "baselines.parquet").column_names == [
        "score_set", "key_a_stereo", "key_b_stereo", "score", "model", "feature_block",
        "p_width", "seed", "config_sha256", "std_version", "licence",
    ]  # fmt: skip


def test_two_runs_give_equal_tables(tmp_path: Path) -> None:
    """Same inputs, same config, same seed: the table is identical (the report differs by time)."""
    validation_path = write_inputs(tmp_path, "validation.parquet")
    out = run(tmp_path, validation_path)
    first = pd.read_parquet(out / "baselines.parquet")
    run(tmp_path, validation_path)
    pd.testing.assert_frame_equal(first, pd.read_parquet(out / "baselines.parquet"))


def test_report_records_the_tie_structure(tmp_path: Path) -> None:
    """The report carries, per model and score set, n_distinct_scores and largest_tie (the number
    of pairs sharing the most common score), plus the config hashes from write_report. B7 on the
    two validation pairs has sizes 8+7 = 15 and 9+10 = 19, so 2 distinct scores."""
    out = run(tmp_path, write_inputs(tmp_path, "validation.parquet"))
    report = json.loads((out / "baselines.report.json").read_text(encoding="utf-8"))
    assert {"n_distinct_scores", "largest_tie", "configs"} <= collect_keys(report)
    assert report["configs"]


def test_validation_columns_hold_no_label() -> None:
    """Static check only (the weaker of the two): the list of columns the script reads from the
    validation file names neither label nor conversion_mean. It would not notice the script reading
    the whole file; test_pairs_reaching_pair_features_carry_no_label does."""
    assert "label" not in VALIDATION_COLUMNS
    assert "conversion_mean" not in VALIDATION_COLUMNS


def test_pairs_reaching_pair_features_carry_no_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property itself: the fixture validation file holds label and conversion_mean, yet no
    frame handed to pair_features has either column. pair_features is wrapped in the module that
    defines it and in the script that imported it by name; the real function still runs."""
    from platepu.score import pairs as pairs_module

    seen: list[pd.DataFrame] = []
    real = pairs_module.pair_features

    def spy(pairs: pd.DataFrame, substrates: pd.DataFrame) -> pd.DataFrame:
        seen.append(pairs)
        return real(pairs, substrates)

    monkeypatch.setattr(pairs_module, "pair_features", spy)
    monkeypatch.setattr(score_module, "pair_features", spy)
    validation_path = write_inputs(tmp_path, "validation.parquet")
    assert {"label", "conversion_mean"} <= set(pd.read_parquet(validation_path).columns)
    run(tmp_path, validation_path)
    assert len(seen) == 2  # validation and inventory
    for frame in seen:
        assert not {"label", "conversion_mean"} & set(frame.columns)


def test_inventory_columns_outside_the_list_do_not_reach_pair_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture inventory holds substrate_a, substrate_b and licence as the real one does. Read
    with the explicit column list they must not reach pair_features; read whole, they would."""
    from platepu.score import pairs as pairs_module

    seen: list[pd.DataFrame] = []
    real = pairs_module.pair_features

    def spy(pairs: pd.DataFrame, substrates: pd.DataFrame) -> pd.DataFrame:
        seen.append(pairs)
        return real(pairs, substrates)

    monkeypatch.setattr(score_module, "pair_features", spy)
    validation_path = write_inputs(tmp_path, "validation.parquet")
    inventory = (
        tmp_path / "private" / "unlabeled" / "amide_coupling" / f"{FAMILY}.parquet"
    )
    assert {"substrate_a", "substrate_b", "licence"} <= set(
        pd.read_parquet(inventory).columns
    )
    run(tmp_path, validation_path)
    inventory_pairs = seen[1]
    assert list(inventory_pairs.columns) == PAIR_COLUMNS
