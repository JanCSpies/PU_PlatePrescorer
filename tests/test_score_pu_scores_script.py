"""Tests for scripts/score_pu_scores.py, the stage S2b entry point, on tiny hand-written rows in a tmp root.

Written BEFORE the script exists (2026-11-02). The fitting and the report are tested in
tests/test_score_pu_scores.py; here only the script's own work is pinned: which files it reads, which
columns, where it writes, what licence the rows carry, what the report says and what it refuses.

Every row is made up (open, hand-written; no Chemlex, Pistachio or inventory row). The substrates are
those of tests/test_score_pu_script.py: "P-like" ones share one fingerprint, "odd" ones another.
Training: 8 positives of width ``all`` (plus 4 of another family the script must not read); spy arm,
an S2a table of 10 rows of which 4 are reliable negatives; naive arm, 8 unlabeled pairs. Scored: 4
validation pairs and 4 inventory pairs, mixed P-like/odd and in neither training class.

The configs are real YAML. The S2b config is passed by ``--config``. The S2a config the spy arm points
at is read by the script from the repository path ``configs/models/<stem>.yml``, relative to the
working directory as ``scripts/score_baselines.py`` reads ``configs/models/baselines.yml``; so every
test runs in ``tmp_path/repo`` (the ``in_repo`` fixture) with that folder created there. That is
awkward and it is the contract: no configs-directory argument is invented.

The script is imported inside ``run`` and not at the top, so that until it exists every test fails
by itself with ModuleNotFoundError instead of one collection error hiding them all.
"""

import importlib
import itertools
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from platepu.io import strictest_licence
from platepu.score.pu_config import (
    PuScoreConfig,
    config_sha256,
    load_pu_config,
    load_pu_score_config,
)

CLASS = "amide_coupling"
FAMILY = "HATU_DIPEA_DMF"
OTHER_FAMILY = "other_family"
STD = "std1"
N_BITS = 8
SPY_STEM = "pu_test_spy"
P_LIKE_ACID = [1, 1, 0, 0, 0, 0, 0, 0]
P_LIKE_AMINE = [0, 0, 1, 1, 0, 0, 0, 0]
ODD_ACID = [0, 0, 0, 0, 1, 1, 0, 0]
ODD_AMINE = [0, 0, 0, 0, 0, 0, 1, 1]
COLUMNS = [
    "score_set", "key_a_stereo", "key_b_stereo", "score", "model", "feature_block", "p_width",
    "seed", "config_sha256", "std_version", "licence",
]  # fmt: skip

SPY_YAML = """\
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
SCORE_YAML = """\
model: {model}
feature_block: {block}
p_width: all
unlabeled: u_stem
negatives: {negatives}
learner: HistGradientBoostingClassifier
learner_params:
  max_iter: 20
  early_stopping: false
  min_samples_leaf: 1
seed: 7
"""


@pytest.fixture(autouse=True)
def in_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Work in tmp_path/repo, where configs/models/ is, because the script reads the S2a config by
    the repository path ``configs/models/<stem>.yml``."""
    (tmp_path / "repo" / "configs" / "models").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "repo")


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


def substrate_table(std_version: str) -> pa.Table:
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
            "std_version": std_version,
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


def with_licence(frame: pd.DataFrame, licence: str) -> pd.DataFrame:
    """All rows open except the first, which carries ``licence``."""
    frame = frame.assign(licence="open")
    frame.loc[0, "licence"] = licence
    return frame


def write_spy_config(block: str = "morgan_centre") -> Path:
    """The S2a config the spy arm points at, at the repository path the script reads."""
    path = Path("configs/models") / f"{SPY_STEM}.yml"
    path.write_text(SPY_YAML.format(block=block), encoding="utf-8")
    return path


def write_score_config(
    directory: Path, arm: str, block: str = "morgan_centre", name: str | None = None
) -> Path:
    """A real stage S2b YAML in ``directory``; its stem names the output."""
    negatives = SPY_STEM if arm == "spy" else "null"
    path = directory / (name or f"pu_score_test_{arm}.yml")
    path.write_text(
        SCORE_YAML.format(model=f"pair_hgb_{arm}", block=block, negatives=negatives),
        encoding="utf-8",
    )
    return path


def write_inputs(
    root: Path,
    validation_dir: Path,
    arm: str = "spy",
    *,
    positives_licence: str = "open",
    negatives_licence: str = "open",
    unlabeled_licence: str = "open",
    validation_licence: str = "open",
    inventory_licence: str = "open",
    std: dict[str, str] | None = None,
    spy_block: str = "morgan_centre",
    table_cut_seed: int | None = None,
    negatives_stem: str = SPY_STEM,
    extra_validation: dict[str, list[object]] | None = None,
    extra_pair: tuple[str, str] | None = None,
    extra_pair_in: str = "validation",
) -> Path:
    """Write every input of one arm at the designed paths; return the validation file.

    ``std`` overrides the std_version of one table by name. The spy arm gets the S2a config and
    table (no unlabeled table: it must not need one); the naive arm gets the unlabeled table and no
    ``pu/`` folder at all. ``table_cut_seed`` hashes the S2a table under a config of another seed.
    """
    std = {"positives": STD, "negatives": STD, "unlabeled": STD, "substrates": STD,
           "validation": STD, **(std or {})}  # fmt: skip
    p_like = list(
        itertools.product([f"PA{i}" for i in range(4)], [f"PN{i}" for i in range(4)])
    )
    odd = list(itertools.product(["OA0", "OA1"], ["ON0", "ON1"]))
    wanted = pair_table(p_like[:8]).assign(family="all", n_reports=1)
    unwanted = pair_table(p_like[8:12]).assign(family=OTHER_FAMILY, n_reports=1)
    positives = pd.concat([wanted, unwanted], ignore_index=True)
    positives = with_licence(positives, positives_licence).assign(
        std_version=std["positives"]
    )
    features_dir = root / "private" / "features" / CLASS
    for folder in (
        root / "positives" / CLASS,
        features_dir,
        root / "private" / "unlabeled" / CLASS,
        validation_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    positives.to_parquet(root / "positives" / CLASS / "positives.parquet")
    pq.write_table(
        substrate_table(std["substrates"]), features_dir / "substrates.parquet"
    )

    scored = [("PA2", "ON0"), ("PA3", "ON1"), ("OA0", "PN2"), ("OA1", "PN3")]
    extra = [extra_pair] if extra_pair else []
    validation_pairs = scored + (extra if extra_pair_in == "validation" else [])
    validation = with_licence(pair_table(validation_pairs), validation_licence)
    validation = validation.assign(std_version=std["validation"])
    for column, values in (extra_validation or {}).items():
        validation[column] = values
    validation.to_parquet(validation_dir / "validation.parquet")
    inventory_pairs = [("PA2", "ON1"), ("PA3", "ON0"), ("OA0", "PN3"), ("OA1", "PN2")]
    inventory = pair_table(
        inventory_pairs + (extra if extra_pair_in == "inventory" else [])
    )
    with_licence(inventory, inventory_licence).to_parquet(
        root / "private" / "unlabeled" / CLASS / f"{FAMILY}.parquet"
    )

    if arm == "naive":
        unlabeled = with_licence(pair_table(p_like[12:16] + odd), unlabeled_licence)
        unlabeled = unlabeled.assign(n_draws=1, std_version=std["unlabeled"])
        (root / "unlabeled" / CLASS).mkdir(parents=True)
        unlabeled.to_parquet(root / "unlabeled" / CLASS / "u_stem.parquet")
        return validation_dir / "validation.parquet"

    spy_config = load_pu_config(write_spy_config(spy_block))
    if table_cut_seed is not None:
        spy_config = replace(spy_config, seed=table_cut_seed)
    reliable = pair_table(odd).assign(in_p=False, reliable_negative=True, weight=2.0)
    unreliable = pair_table(p_like[12:16]).assign(
        in_p=False, reliable_negative=False, weight=1.0
    )
    spies = pair_table(p_like[:2]).assign(
        in_p=True, reliable_negative=False, weight=1.0
    )
    negatives = pd.concat([reliable, unreliable, spies], ignore_index=True)
    negatives = with_licence(negatives, negatives_licence).assign(
        fold=0, spy_score=0.5, threshold=0.4, model="pair_hgb", feature_block=spy_block, seed=7,
        config_sha256=config_sha256(spy_config), std_version=std["negatives"],
    )  # fmt: skip
    (root / "pu" / CLASS).mkdir(parents=True)
    negatives.to_parquet(root / "pu" / CLASS / f"{negatives_stem}.negatives.parquet")
    return validation_dir / "validation.parquet"


def run(root: Path, config: Path, validation: Path) -> Path:
    """Run the script (imported here, see the module docstring); return the folder it writes to."""
    module = importlib.import_module("score_pu_scores")
    module.main(
        [
            CLASS,
            FAMILY,
            str(root),
            "--config",
            str(config),
            "--validation",
            str(validation),
        ]
    )
    return root / "private" / "scores" / CLASS / FAMILY


def setup_and_run(
    tmp_path: Path, arm: str = "spy", **options: object
) -> tuple[Path, Path, Path]:
    """Inputs plus config, then the run: ``(root, folder written to, config)``."""
    root = tmp_path / "derived"
    validation = write_inputs(root, tmp_path / "benchmark", arm, **options)  # type: ignore[arg-type]
    config = write_score_config(tmp_path, arm)
    return root, run(root, config, validation), config


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_spy_arm_lands_at_the_designed_path_with_stage_s1_columns(
    tmp_path: Path,
) -> None:
    """The table is private/scores/<class>/<family>/<config stem>.parquet with S1's eleven columns in
    S1's order; 4 validation + 4 inventory rows; float32 score; the spy model name; a feature block
    on every row; a valid report beside it; and the spy arm read no unlabeled table (none exists)."""
    root, out, config = setup_and_run(tmp_path, "spy")
    table = pq.read_table(out / f"{config.stem}.parquet")
    assert table.column_names == COLUMNS
    assert table.schema.field("score").type == pa.float32()
    frame = table.to_pandas()
    assert frame["score_set"].value_counts().to_dict() == {
        "validation": 4,
        "inventory": 4,
    }
    assert set(frame["model"]) == {"pair_hgb_spy"}
    assert set(frame["feature_block"]) == {"morgan_centre"}
    assert set(frame["p_width"]) == {"all"}
    assert frame["feature_block"].notna().all()
    assert not (root / "unlabeled").exists()
    assert isinstance(read_report(out / f"{config.stem}.report.json"), dict)


def test_naive_arm_runs_in_a_tree_with_no_pu_directory(tmp_path: Path) -> None:
    """The arm that answers 'what did the spy step buy?' must run on its own: no pu/ folder anywhere,
    no S2a config for it to find, and the rows say pair_hgb_naive."""
    root, out, config = setup_and_run(tmp_path, "naive")
    assert not (root / "pu").exists()
    assert not list(Path("configs/models").iterdir())
    frame = pd.read_parquet(out / f"{config.stem}.parquet")
    assert list(frame.columns) == COLUMNS
    assert len(frame) == 8
    assert set(frame["model"]) == {"pair_hgb_naive"}


def test_negatives_cut_under_another_s2a_config_are_refused(tmp_path: Path) -> None:
    """The table's config_sha256 is that of an S2a config with seed 99, the YAML on disk says seed 7:
    the script stops on config_sha256 and writes nothing."""
    root = tmp_path / "derived"
    validation = write_inputs(root, tmp_path / "benchmark", table_cut_seed=99)
    with pytest.raises(ValueError, match="config_sha256"):
        run(root, write_score_config(tmp_path, "spy"), validation)
    assert not (root / "private" / "scores").exists()


def test_an_s2a_config_that_disagrees_on_feature_block_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    """The S2a YAML (and the table's hash, which matches it) is on morgan, this run on morgan_centre:
    the error names feature_block, so the model is never fitted on a selection made for another."""
    root = tmp_path / "derived"
    validation = write_inputs(root, tmp_path / "benchmark", spy_block="morgan")
    with pytest.raises(ValueError, match="feature_block"):
        run(root, write_score_config(tmp_path, "spy", "morgan_centre"), validation)


def test_a_negatives_stem_with_no_s2a_config_raises_with_the_path(
    tmp_path: Path,
) -> None:
    """config.negatives names no_such_spy; its table exists, configs/models/no_such_spy.yml does not.
    The error carries the path it looked for."""
    root = tmp_path / "derived"
    validation = write_inputs(
        root, tmp_path / "benchmark", negatives_stem="no_such_spy"
    )
    config = write_score_config(tmp_path, "spy")
    config.write_text(
        config.read_text(encoding="utf-8").replace(SPY_STEM, "no_such_spy"),
        encoding="utf-8",
    )
    missing = "configs/models/no_such_spy.yml"
    with pytest.raises((FileNotFoundError, ValueError), match=re.escape(missing)):
        run(root, config, validation)


def test_validation_must_be_named_validation_parquet(tmp_path: Path) -> None:
    """As in stage S1: a benchmark file with any other name (labels, test) is refused before reading."""
    root = tmp_path / "derived"
    validation = write_inputs(root, tmp_path / "benchmark")
    renamed = validation.with_name("test.parquet")
    validation.rename(renamed)
    with pytest.raises(ValueError, match=r"validation\.parquet"):
        run(root, write_score_config(tmp_path, "spy"), renamed)


def test_validation_is_required(tmp_path: Path) -> None:
    """No --validation, no run: argparse exits with status 2."""
    root = tmp_path / "derived"
    write_inputs(root, tmp_path / "benchmark")
    config = write_score_config(tmp_path, "spy")
    module = importlib.import_module("score_pu_scores")
    with pytest.raises(SystemExit) as refused:
        module.main([CLASS, FAMILY, str(root), "--config", str(config)])
    assert refused.value.code == 2


@pytest.mark.parametrize("arm", ["spy", "naive"])
def test_a_label_cannot_reach_the_model_or_the_output(tmp_path: Path, arm: str) -> None:
    """Two runs whose validation files differ ONLY in the values of label and conversion_mean give
    identical scores, and neither column is in the output table or the report. A column the step
    never asked for cannot change its answer."""
    results = []
    for name, labels in (("one", [0, 1, 0, 1]), ("two", [1, 1, 1, 0])):
        extra = {
            "label": labels,
            "conversion_mean": [5.0 * v + i for i, v in enumerate(labels)],
        }
        root = tmp_path / name
        root.mkdir()
        validation = write_inputs(
            root / "derived", root / "benchmark", arm, extra_validation=extra
        )
        assert {"label", "conversion_mean"} <= set(
            pq.read_table(validation).column_names
        )
        config = write_score_config(root, arm)
        out = run(root / "derived", config, validation)
        results.append((out, config.stem))
    first, second = (pd.read_parquet(out / f"{stem}.parquet") for out, stem in results)
    pd.testing.assert_frame_equal(first, second)
    assert not {"label", "conversion_mean"} & set(first.columns)
    for out, stem in results:
        text = (out / f"{stem}.report.json").read_text(encoding="utf-8")
        assert "conversion_mean" not in text
        assert '"label"' not in text


def test_licence_is_per_scored_set_and_the_file_lands_under_private(
    tmp_path: Path,
) -> None:
    """Noncommercial validation, internal inventory, open training: each set carries
    io.strictest_licence of its own licences (and of the open training), so the two differ inside one
    table. The file is written under private/ and nowhere else."""
    root, out, config = setup_and_run(
        tmp_path,
        "spy",
        validation_licence="noncommercial",
        inventory_licence="internal",
    )
    frame = pd.read_parquet(out / f"{config.stem}.parquet")
    by_set = frame.groupby("score_set")["licence"].agg(set).to_dict()
    assert by_set == {
        "validation": {strictest_licence(["noncommercial", "open"])},
        "inventory": {strictest_licence(["internal", "open"])},
    }
    assert by_set == {"validation": {"noncommercial"}, "inventory": {"internal"}}
    assert out.is_relative_to(root / "private")
    assert not (root / "scores").exists()


def test_open_scored_sets_and_open_training_stay_open(tmp_path: Path) -> None:
    """The baseline of the two tests around it: nothing restricted in, nothing restricted out."""
    _, out, config = setup_and_run(tmp_path, "spy")
    assert set(pd.read_parquet(out / f"{config.stem}.parquet")["licence"]) == {"open"}


@pytest.mark.parametrize("arm", ["spy", "naive"])
def test_a_licensed_positive_widens_both_scored_sets(tmp_path: Path, arm: str) -> None:
    """One 'licensed' positives row, open scored sets: the training side widens BOTH, so validation
    and inventory rows are 'licensed' (and the write lands under private/, or write_table would
    have raised)."""
    _, out, config = setup_and_run(tmp_path, arm, positives_licence="licensed")
    frame = pd.read_parquet(out / f"{config.stem}.parquet")
    assert set(frame["licence"]) == {"licensed"}


def test_a_licensed_positive_over_mixed_scored_sets_follows_the_precedence(
    tmp_path: Path,
) -> None:
    """Licensed positive, noncommercial validation, internal inventory. As io.py orders the
    licences (internal, licensed, noncommercial, open) licensed is stricter than noncommercial and
    internal is stricter than licensed: validation becomes 'licensed', inventory stays 'internal'."""
    _, out, config = setup_and_run(
        tmp_path, "spy", positives_licence="licensed",
        validation_licence="noncommercial", inventory_licence="internal",
    )  # fmt: skip
    frame = pd.read_parquet(out / f"{config.stem}.parquet")
    by_set = frame.groupby("score_set")["licence"].agg(set).to_dict()
    assert by_set == {"validation": {"licensed"}, "inventory": {"internal"}}


def test_a_licensed_reliable_negative_widens_both_scored_sets(tmp_path: Path) -> None:
    """The negatives table is read by the fit, so its licence counts too (row 0 is a reliable one)."""
    _, out, config = setup_and_run(tmp_path, "spy", negatives_licence="licensed")
    frame = pd.read_parquet(out / f"{config.stem}.parquet")
    assert set(frame["licence"]) == {"licensed"}


def test_a_licensed_unlabeled_row_widens_both_scored_sets_in_the_naive_arm(
    tmp_path: Path,
) -> None:
    """In the naive arm U is the negative class, so its licence counts."""
    _, out, config = setup_and_run(tmp_path, "naive", unlabeled_licence="internal")
    frame = pd.read_parquet(out / f"{config.stem}.parquet")
    assert set(frame["licence"]) == {"internal"}


@pytest.mark.parametrize("arm", ["spy", "naive"])
def test_two_runs_give_identical_tables_and_reports(tmp_path: Path, arm: str) -> None:
    """Same inputs, same config, same seed, two roots: the tables are equal and the reports agree
    except for written_at."""
    config = write_score_config(tmp_path, arm)
    outs = []
    for name in ("first", "second"):
        root = tmp_path / name
        validation = write_inputs(root, tmp_path / "benchmark", arm)
        outs.append(run(root, config, validation))
    first, second = (pd.read_parquet(out / f"{config.stem}.parquet") for out in outs)
    pd.testing.assert_frame_equal(first, second)
    reports = [read_report(out / f"{config.stem}.report.json") for out in outs]
    for report in reports:
        report.pop("written_at")
    assert reports[0] == reports[1]


def test_the_config_stem_names_the_output(tmp_path: Path) -> None:
    """Two configs that differ only in feature_block write two tables named by their stems, and the
    second run leaves the first table byte for byte as it was."""
    root = tmp_path / "derived"
    validation = write_inputs(root, tmp_path / "benchmark", "naive")
    centre = write_score_config(
        tmp_path, "naive", "morgan_centre", "pu_score_centre.yml"
    )
    plain = write_score_config(tmp_path, "naive", "morgan", "pu_score_plain.yml")
    out = run(root, centre, validation)
    first_bytes = (out / "pu_score_centre.parquet").read_bytes()
    run(root, plain, validation)
    near = write_score_config(tmp_path, "naive", "morgan_near", "pu_score_near.yml")
    run(root, near, validation)
    assert set(pd.read_parquet(out / "pu_score_near.parquet")["feature_block"]) == {
        "morgan_near"
    }
    assert (out / "pu_score_centre.parquet").read_bytes() == first_bytes
    assert set(pd.read_parquet(out / "pu_score_centre.parquet")["feature_block"]) == {
        "morgan_centre"
    }
    assert set(pd.read_parquet(out / "pu_score_plain.parquet")["feature_block"]) == {
        "morgan"
    }
    assert (out / "pu_score_centre.report.json").exists()
    assert (out / "pu_score_plain.report.json").exists()


def test_the_table_hash_is_the_value_hash_and_the_report_hashes_the_spy_config(
    tmp_path: Path,
) -> None:
    """config_sha256 on every row is pu_config.config_sha256 of the S2b record (values, as in S2a);
    the report's spy_config_sha256 is that of the S2a record the pointer was checked against."""
    _, out, config = setup_and_run(tmp_path, "spy")
    frame = pd.read_parquet(out / f"{config.stem}.parquet")
    report = read_report(out / f"{config.stem}.report.json")
    expected: PuScoreConfig = load_pu_score_config(config)
    assert set(frame["config_sha256"]) == {config_sha256(expected)}
    assert set(frame["seed"]) == {7}
    assert set(frame["std_version"]) == {STD}
    spy_path = Path("configs/models") / f"{SPY_STEM}.yml"
    assert report["spy_config_sha256"] == config_sha256(load_pu_config(spy_path))


@pytest.mark.parametrize("arm", ["spy", "naive"])
def test_report_counts_match_the_table(tmp_path: Path, arm: str) -> None:
    """rows_out is the table's length (8); rows_in holds each input's row count as the script reads
    it (8 positives of width 'all', not the 12 in the file; the whole negatives table, 10 rows, in
    the spy arm; U, 8 rows, in the naive arm); n_positives and n_negatives are the training rows
    (spy: 8 and the 4 reliable ones; naive: 8 and all 8 of U)."""
    _, out, config = setup_and_run(tmp_path, arm)
    frame = pd.read_parquet(out / f"{config.stem}.parquet")
    report = read_report(out / f"{config.stem}.report.json")
    training = {"negatives": 10} if arm == "spy" else {"unlabeled": 8}
    assert report["rows_out"] == len(frame) == 8
    assert report["rows_in"] == {
        "positives": 8, **training, "substrates": 12, "validation": 4, "inventory": 4,
    }  # fmt: skip
    assert report["n_positives"] == 8
    assert report["n_negatives"] == (4 if arm == "spy" else 8)
    assert report["stage"] and report["class"] == CLASS and report["family"] == FAMILY
    assert report["config"] == str(config)
    assert report["std_version"] == STD
    assert report["negatives"] == (SPY_STEM if arm == "spy" else None)
    assert (report["spy_config_sha256"] is None) == (arm == "naive")
    assert str(config) in report["configs"]


@pytest.mark.parametrize(
    ("arm", "relative"),
    [
        ("spy", f"positives/{CLASS}/positives.parquet"),
        ("spy", f"pu/{CLASS}/{SPY_STEM}.negatives.parquet"),
        ("naive", f"unlabeled/{CLASS}/u_stem.parquet"),
        ("spy", f"private/unlabeled/{CLASS}/{FAMILY}.parquet"),
        ("spy", f"private/features/{CLASS}/substrates.parquet"),
    ],
)
def test_a_missing_input_raises_naming_it(
    tmp_path: Path, arm: str, relative: str
) -> None:
    """Delete one input; the error carries the path that was looked for, so nobody guesses which of
    five inputs is missing."""
    root = tmp_path / "derived"
    validation = write_inputs(root, tmp_path / "benchmark", arm)
    config = write_score_config(tmp_path, arm)
    missing = root / relative
    missing.unlink()
    with pytest.raises((FileNotFoundError, ValueError), match=re.escape(str(missing))):
        run(root, config, validation)


@pytest.mark.parametrize("where", ["validation", "inventory"])
def test_a_scored_pair_missing_from_the_feature_table_raises(
    tmp_path: Path, where: str
) -> None:
    """A validation pair (or an inventory pair) whose substrate is not in the feature table stops the
    run naming the key; coverage was counted at 100 %, so a miss is input we have not seen."""
    root = tmp_path / "derived"
    validation = write_inputs(
        root, tmp_path / "benchmark", extra_pair=("PA0", "NO_SUCH"), extra_pair_in=where
    )
    with pytest.raises(ValueError, match="NO_SUCH"):
        run(root, write_score_config(tmp_path, "spy"), validation)
    assert not (root / "private" / "scores").exists()


@pytest.mark.parametrize(
    ("arm", "where"),
    [
        ("spy", "positives"),
        ("spy", "negatives"),
        ("naive", "unlabeled"),
        ("spy", "substrates"),
        ("spy", "validation"),
    ],
)
def test_a_mixed_std_version_raises(tmp_path: Path, arm: str, where: str) -> None:
    """One standardisation for P, the negatives (or U), the features and the validation pairs
    (CLAUDE.md §6): any one of them on std2 stops the run naming std_version; nothing is written."""
    root = tmp_path / "derived"
    validation = write_inputs(root, tmp_path / "benchmark", arm, std={where: "std2"})
    with pytest.raises(ValueError, match="std_version"):
        run(root, write_score_config(tmp_path, arm), validation)
    assert not (root / "private" / "scores").exists()
