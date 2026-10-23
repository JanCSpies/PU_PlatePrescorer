"""Tests for scripts/map_missing.py: splitting the reactions to map into shards."""

from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from map_missing import write_shards
from map_shard import main as map_shard


def test_shards_hold_at_most_size_rows(tmp_path: Path) -> None:
    """Five reactions with size 2 give three shards (2, 2, 1), numbered from 0."""
    items = [(f"h{i}", f"C{i}>>C") for i in range(5)]
    assert write_shards(items, tmp_path / "cache", 2, root=tmp_path) == 3
    shards = sorted((tmp_path / "cache" / "todo").glob("shard-*.parquet"))
    assert [pq.read_table(s).num_rows for s in shards] == [2, 2, 1]
    assert shards[0].name == "shard-00000.parquet"


def test_nothing_to_map_writes_nothing(tmp_path: Path) -> None:
    """An up-to-date cache gives no shard and no todo folder."""
    assert write_shards([], tmp_path / "cache", 2, root=tmp_path) == 0
    assert not (tmp_path / "cache" / "todo").exists()


def _shard(tmp_path: Path, rsmis: list[str], number: int = 0) -> Path:
    """A shard of the given reactions in ``<tmp>/cache/rxnmapper-test/todo/``."""
    shard = (
        tmp_path / "cache" / "rxnmapper-test" / "todo" / f"shard-{number:05d}.parquet"
    )
    shard.parent.mkdir(parents=True, exist_ok=True)
    hashes = [f"h{number}-{i}" for i in range(len(rsmis))]
    pq.write_table(pa.table({"rsmi_hash": hashes, "rsmi": rsmis}), shard)
    return shard


def _run(tmp_path: Path, shard: Path, python: Path) -> None:
    """map_shard.py on ``shard`` with a paths file that points at ``python`` as the RXNMapper env."""
    paths = tmp_path / "paths.yml"
    paths.write_text(
        f"root: {tmp_path}\nrxnmapper_python: {python}\nmapper_batch_size: 4\n"
    )
    map_shard([str(shard), "--paths", str(paths)])


def _part(tmp_path: Path, number: int) -> Path:
    return tmp_path / "cache" / "rxnmapper-test" / f"part-{number:05d}.parquet"


def test_second_run_does_not_overwrite_an_earlier_part(
    tmp_path: Path, stub_rxnmapper: Callable[[str], Path]
) -> None:
    """map_missing numbers shards from 00000 every run, so a new source (or the full corpus after
    the sample) produced shard-00000 again and replaced the earlier part-00000: thousands of cached
    mappings vanished with no message and RXNMapper was paid for again. Now it must stop, leaving
    the old part byte for byte and the shard in place."""
    old = _part(tmp_path, 0)
    old.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "rsmi_hash": ["old"],
                "rsmi": ["C>>C"],
                "mapped_rsmi": ["[C:1]"],
                "confidence": [0.9],
            }
        ),
        old,
    )
    before = old.read_bytes()
    shard = _shard(tmp_path, ["CC>>CO"], number=0)
    with pytest.raises((FileExistsError, ValueError)):
        _run(tmp_path, shard, stub_rxnmapper("ok"))
    assert old.read_bytes() == before
    assert shard.exists()


def test_fresh_part_name_is_written_beside_the_earlier_part(
    tmp_path: Path, stub_rxnmapper: Callable[[str], Path]
) -> None:
    """Control: shard-00001 becomes part-00001, the shard is removed, part-00000 is untouched."""
    first = _part(tmp_path, 0)
    first.parent.mkdir(parents=True)
    first.write_bytes(b"earlier part")
    shard = _shard(tmp_path, ["CC>>CO", "CCC>>CCO"], number=1)
    _run(tmp_path, shard, stub_rxnmapper("ok"))
    assert pq.read_table(_part(tmp_path, 1)).num_rows == 2
    assert first.read_bytes() == b"earlier part"
    assert not shard.exists()


HUNDRED = [f"{'C' * (i + 1)}>>{'C' * (i + 1)}O" for i in range(100)]


@pytest.mark.parametrize("failing", [100, 50])
def test_shard_where_most_reactions_failed_is_not_cached(
    tmp_path: Path, stub_rxnmapper: Callable[[str], Path], failing: int
) -> None:
    """rxnutils reports every failure of RXNMapper, an exhausted memory or a broken environment
    included, as 'input back, confidence 0'. Cached as mapped_rsmi = None, those reactions are never
    retried, because the cache keys on the hash alone: a run printed '100 mapped, 100 failed' and the
    positives were gone for good. On 63 real shards (625,539 ORDerly reactions) the worst shard had
    0.03 % failures, so a shard with more than 5 % failed is an environment fault: raise, write
    nothing, keep the shard so it can be mapped again."""
    shard = _shard(tmp_path, HUNDRED)
    with pytest.raises(Exception, match="fail"):
        _run(tmp_path, shard, stub_rxnmapper(f"failing_first:{failing}"))
    assert not _part(tmp_path, 0).exists()
    assert shard.exists()


def test_shard_with_a_few_failed_reactions_is_written(
    tmp_path: Path, stub_rxnmapper: Callable[[str], Path]
) -> None:
    """Control: 2 of 100 failed (2 %, about 60 times the worst real shard) is a normal shard; it is
    written with the two failures cached as None."""
    shard = _shard(tmp_path, HUNDRED)
    _run(tmp_path, shard, stub_rxnmapper("failing_first:2"))
    part = pq.read_table(_part(tmp_path, 0))
    assert (part.num_rows, part["mapped_rsmi"].null_count) == (100, 2)
    assert not shard.exists()
