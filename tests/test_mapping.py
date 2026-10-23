"""Tests for the mapping cache and stage 3 (docs/design/rebuild.md, "Mapping"), with a fake mapper.

The real RXNMapper runs in its own environment; ``test_real_rxnmapper`` uses it only when
``RXNMAPPER_PYTHON`` points to that environment's interpreter.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from platepu.reactions.mapping import (
    join_mapped,
    map_reactions,
    read_cache,
    rxnmapper,
    to_map,
)

AMIDE = "CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1"
MAPPED_AMIDE = (
    "[CH3:1][C:2](=[O:3])O.[NH2:4][CH2:5][c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1>>"
    "[CH3:1][C:2](=[O:3])[NH:4][CH2:5][c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
)
BROKEN = "CC>>CO"


def fake_mapper(rsmis: Sequence[str]) -> list[tuple[str, float]]:
    """Maps the amide; 'fails' on anything else the way rxnutils does (input back, confidence 0)."""
    return [(MAPPED_AMIDE, 0.75) if rsmi == AMIDE else (rsmi, 0.0) for rsmi in rsmis]


def _rsmi(rows: list[tuple[str, str, str, str]]) -> pa.Table:
    """A stage-2 table from (record_id, licence, rsmi, rsmi_hash) tuples."""
    names = ["record_id", "licence", "rsmi", "rsmi_hash"]
    return pa.table({name: [row[i] for row in rows] for i, name in enumerate(names)})


def test_failure_is_kept_as_none_not_as_the_input() -> None:
    """rxnutils returns the unmapped input on failure; the cache stores None and confidence 0."""
    table = map_reactions([("h1", AMIDE), ("h2", BROKEN)], fake_mapper)
    rows = {row["rsmi_hash"]: row for row in table.to_pylist()}
    assert rows["h1"]["mapped_rsmi"] == MAPPED_AMIDE
    assert rows["h2"]["mapped_rsmi"] is None
    assert rows["h2"]["confidence"] == 0.0


def test_to_map_skips_cached_and_repeated_reactions() -> None:
    """Each reaction is mapped once: hashes already in the cache, or seen twice, are not repeated."""
    stage2 = _rsmi(
        [
            ("a:1", "open", AMIDE, "h1"),
            ("b:1", "open", AMIDE, "h1"),
            ("a:2", "open", BROKEN, "h2"),
        ]
    )
    cache = map_reactions([("h1", AMIDE)], fake_mapper)
    assert to_map([stage2], cache) == [("h2", BROKEN)]


def test_cache_is_read_from_all_part_files(tmp_path: Path) -> None:
    """The cache is append-only part files; reading it collects every part."""
    folder = tmp_path / "cache"
    folder.mkdir()
    pq.write_table(
        map_reactions([("h1", AMIDE)], fake_mapper), folder / "part-0.parquet"
    )
    pq.write_table(
        map_reactions([("h2", BROKEN)], fake_mapper), folder / "part-1.parquet"
    )
    assert sorted(read_cache([folder])["rsmi_hash"].to_pylist()) == ["h1", "h2"]


def test_empty_cache_folder_gives_an_empty_cache(tmp_path: Path) -> None:
    """Before the first mapping run there is nothing cached, and that is not an error."""
    assert read_cache([tmp_path / "missing"]).num_rows == 0


def test_join_keeps_mapped_and_rejects_failed() -> None:
    """Stage 3: mapped reactions go on; failed mappings are rejected with a reason."""
    stage2 = _rsmi([("a:1", "open", AMIDE, "h1"), ("a:2", "open", BROKEN, "h2")])
    cache = map_reactions([("h1", AMIDE), ("h2", BROKEN)], fake_mapper)
    table, rejected = join_mapped(stage2, cache)
    assert table["record_id"].to_pylist() == ["a:1"]
    assert table["mapped_rsmi"].to_pylist() == [MAPPED_AMIDE]
    assert rejected.to_pylist() == [{"record_id": "a:2", "reason": "mapping_failed"}]


def test_join_stops_when_a_reaction_was_never_mapped() -> None:
    """A reaction missing from the cache means the mapping step did not run; never skip it silently."""
    stage2 = _rsmi([("a:1", "open", AMIDE, "h1")])
    with pytest.raises(ValueError, match="not mapped"):
        join_mapped(stage2, map_reactions([], fake_mapper))


@pytest.mark.skipif(
    "RXNMAPPER_PYTHON" not in os.environ, reason="RXNMapper env not set"
)
def test_real_rxnmapper() -> None:
    """The real mapper, through rxnutils, maps the amide and fails on a broken reaction."""
    mapper = rxnmapper(Path(os.environ["RXNMAPPER_PYTHON"]), batch_size=8)
    (mapped, confidence), (_, zero) = mapper([AMIDE, "F[P-](F)(F)(F)(F)F.xx>>CC"])
    assert ":" in mapped and confidence > 0.5
    assert zero == 0.0


THREE = [
    "CC>>CO",
    "CCC>>CCO",
    "CCCC>>CCCO",
]  # distinct reactions; the stub mapper does not parse them


def test_mapper_output_in_input_order_is_accepted(
    stub_rxnmapper: Callable[[str], Path],
) -> None:
    """Control for the two tests below: the stub in order gives one result per reaction, in order."""
    results = rxnmapper(stub_rxnmapper("ok"), batch_size=4)(THREE)
    assert [mapped for mapped, _ in results] == ["[CH3:1]" + rsmi for rsmi in THREE]


@pytest.mark.parametrize("mode", ["reversed", "padded"])
def test_mapper_output_not_matching_the_input_rows_stops_the_run(
    stub_rxnmapper: Callable[[str], Path], mode: str
) -> None:
    """Results are matched to reactions by position, so a mapper that returns the right number of rows
    in another order (``reversed``), or drops one and pads with a duplicate (``padded``), must stop
    the run. Before this, reaction X's mapping was cached under reaction Y's hash, permanently, and
    flowed into the classes and the positives; only the row count was checked."""
    with pytest.raises(ValueError):
        rxnmapper(stub_rxnmapper(mode), batch_size=4)(THREE)


def test_failing_mapper_process_reports_its_stderr(
    stub_rxnmapper: Callable[[str], Path],
) -> None:
    """A missing rxnmapper in the target environment said only 'returned non-zero exit status 1',
    because CalledProcessError.__str__ leaves out stderr and nobody printed it; the cause was lost."""
    mapper = rxnmapper(stub_rxnmapper("error"), batch_size=4)
    with pytest.raises(Exception, match="No module named 'rxnmapper'"):
        mapper(THREE)


def test_mapper_process_has_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a timeout a hung mapper hangs a cluster job until the scheduler kills it, with no
    message; the subprocess call must carry one."""
    seen: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise RuntimeError("stop before the output is read")

    monkeypatch.setattr("platepu.reactions.mapping.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="stop before"):
        rxnmapper(Path("python"), batch_size=4)(THREE)
    assert seen.get("timeout") is not None
