"""Tests for load_baselines_config on tiny YAML files in tmp_path, plus the real
configs/models/baselines.yml (read from the repository root, as the script does).
"""

from pathlib import Path

import pytest

from platepu.score.baselines import load_baselines_config

REAL = Path("configs/models/baselines.yml")
SCORES = "amine_subclass_scores:\n  aniline: 1\n  other: 2.5\n"


def write(tmp_path: Path, text: str) -> Path:
    """Write a model YAML and return its path."""
    path = tmp_path / "baselines.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_real_baselines_config_loads() -> None:
    """Seed 20261021 and the five named amine subclasses, `other` among them."""
    config = load_baselines_config(REAL)
    assert config.seed == 20261021
    assert set(config.amine_subclass_scores) == {
        "heteroaryl_amine", "aniline", "aliphatic_secondary", "aliphatic_primary", "other",
    }  # fmt: skip


def test_scores_become_floats_keyed_by_string(tmp_path: Path) -> None:
    """YAML integer 1 and float 2.5 both arrive as float."""
    config = load_baselines_config(write(tmp_path, f"seed: 7\n{SCORES}"))
    assert config.amine_subclass_scores == {"aniline": 1.0, "other": 2.5}
    assert all(type(v) is float for v in config.amine_subclass_scores.values())


def test_missing_seed_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, SCORES)
    with pytest.raises(ValueError, match="baselines.yml"):
        load_baselines_config(path)


def test_missing_amine_subclass_scores_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, "seed: 7\n")
    with pytest.raises(ValueError, match="baselines.yml"):
        load_baselines_config(path)


def test_seed_that_is_not_a_number_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, f"seed: twelve\n{SCORES}")
    with pytest.raises(ValueError, match="baselines.yml"):
        load_baselines_config(path)


def test_scores_given_as_a_list_name_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, "seed: 7\namine_subclass_scores: [1, 2]\n")
    with pytest.raises(ValueError, match="baselines.yml"):
        load_baselines_config(path)


def test_seed_given_as_a_numeric_string_is_coerced(tmp_path: Path) -> None:
    """Pinned on purpose: coercion with int(), as load_feature_config does, not an isinstance
    check. The quoted '20261021' becomes the integer 20261021."""
    config = load_baselines_config(write(tmp_path, f'seed: "20261021"\n{SCORES}'))
    assert config.seed == 20261021
    assert type(config.seed) is int
