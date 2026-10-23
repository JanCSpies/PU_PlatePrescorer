"""Atom mapping with RXNMapper, the mapping cache, and stage 3 (docs/design/rebuild.md, "Mapping").

RXNMapper runs in its own environment (torch, transformers); ``rxnmapper()`` calls rxnutils' mapping
script there and returns a plain function, so everything else is tested with a fake mapper. The cache
is append-only parquet part files keyed by ``rsmi_hash``; a failed mapping is cached too (as None), so
it is not retried on every run.
"""

import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from platepu.io import REJECTED_SCHEMA

# rsmis -> one (mapped reaction SMILES, confidence) per rsmi, in the same order
Mapper = Callable[[Sequence[str]], list[tuple[str, float]]]

# Wall-clock limit for one RXNMapper batch. Without it a hung mapper holds a cluster array job
# open for its whole allocation; 30 minutes is far above the minutes a batch of 16 takes.
TIMEOUT = 1800
CACHE_SCHEMA = pa.schema(
    [
        ("rsmi_hash", pa.string()),
        ("rsmi", pa.string()),
        ("mapped_rsmi", pa.string()),
        ("confidence", pa.float64()),
    ]
)
MAPPED_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("licence", pa.string()),
        ("rsmi_hash", pa.string()),
        ("mapped_rsmi", pa.string()),
        ("confidence", pa.float64()),
    ]
)


def map_reactions(items: Sequence[tuple[str, str]], mapper: Mapper) -> pa.Table:
    """Map ``(rsmi_hash, rsmi)`` pairs into cache rows.

    A mapping counts as failed when the confidence is 0 or the output has no atom map (rxnutils
    then returns the input unchanged); failed rows get ``mapped_rsmi = None`` and confidence 0.
    """
    results = mapper([rsmi for _, rsmi in items]) if items else []
    if len(results) != len(items):
        raise ValueError(
            f"mapper returned {len(results)} results for {len(items)} reactions"
        )
    rows = []
    for (rsmi_hash, rsmi), (mapped, confidence) in zip(items, results, strict=True):
        ok = confidence > 0 and ":" in mapped
        rows.append(
            {
                "rsmi_hash": rsmi_hash,
                "rsmi": rsmi,
                "mapped_rsmi": mapped if ok else None,
                "confidence": confidence if ok else 0.0,
            }
        )
    return pa.Table.from_pylist(rows, schema=CACHE_SCHEMA)


def read_cache(folders: Iterable[Path]) -> pa.Table:
    """All cached mappings from the ``part-*.parquet`` files in the given folders."""
    parts = [
        part for folder in folders for part in sorted(folder.glob("part-*.parquet"))
    ]
    if not parts:
        return CACHE_SCHEMA.empty_table()
    return pa.concat_tables(pq.read_table(part, schema=CACHE_SCHEMA) for part in parts)


def to_map(stage2_tables: Iterable[pa.Table], cache: pa.Table) -> list[tuple[str, str]]:
    """The ``(rsmi_hash, rsmi)`` pairs not yet in the cache, each once, in first-seen order."""
    done = set(cache["rsmi_hash"].to_pylist())
    missing: dict[str, str] = {}
    for table in stage2_tables:
        for rsmi_hash, rsmi in zip(
            table["rsmi_hash"].to_pylist(), table["rsmi"].to_pylist(), strict=True
        ):
            if rsmi_hash not in done:
                missing.setdefault(rsmi_hash, rsmi)
    return list(missing.items())


def join_mapped(stage2: pa.Table, cache: pa.Table) -> tuple[pa.Table, pa.Table]:
    """Stage 3: attach each record's mapping; failed mappings become rejects (``mapping_failed``).

    Raises:
        ValueError: if a reaction is missing from the cache, i.e. the mapping step has not run.
    """
    lookup = {
        row["rsmi_hash"]: (row["mapped_rsmi"], row["confidence"])
        for row in cache.to_pylist()
    }
    unmapped = [h for h in stage2["rsmi_hash"].to_pylist() if h not in lookup]
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} reactions not mapped yet; run the mapping step first"
        )
    rows, rejected = [], []
    for row in stage2.to_pylist():
        mapped, confidence = lookup[row["rsmi_hash"]]
        if mapped is None:
            rejected.append({"record_id": row["record_id"], "reason": "mapping_failed"})
            continue
        rows.append({**row, "mapped_rsmi": mapped, "confidence": confidence})
    return (
        pa.Table.from_pylist(rows, schema=MAPPED_SCHEMA),
        pa.Table.from_pylist(rejected, schema=REJECTED_SCHEMA),
    )


def rxnmapper(python: Path, batch_size: int) -> Mapper:
    """A mapper that runs rxnutils' RXNMapper script with the interpreter of the RXNMapper env."""

    def run(rsmis: Sequence[str]) -> list[tuple[str, float]]:
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in.tsv", Path(tmp) / "out.tsv"
            pd.DataFrame({"rsmi": list(rsmis)}).to_csv(source, sep="\t", index=False)
            command = [str(python), "-W", "ignore", "-m", "rxnutils.data.mapping"]
            options = [
                "--input",
                str(source),
                "--output",
                str(target),
                "--column",
                "rsmi",
            ]
            batch = ["--mapper_batch_size", str(batch_size)]
            try:
                subprocess.run(
                    [*command, *options, *batch],
                    check=True,
                    capture_output=True,
                    timeout=TIMEOUT,
                )
            except subprocess.CalledProcessError as error:
                # CalledProcessError does not print stderr, so the real cause -- usually
                # "you need an environment where rxnmapper is installed" -- was being thrown away
                # and the operator saw only "returned non-zero exit status 1".
                raise RuntimeError(
                    f"the RXNMapper interpreter {python} failed: "
                    f"{error.stderr.decode(errors='replace').strip()}"
                ) from error
            out = pd.read_csv(target, sep="\t", keep_default_na=False)
        # Results are matched to inputs by position, so a reordered, deduplicated or padded output
        # would attach one reaction's mapping to another's hash -- permanently, because the cache
        # keys on the hash, and invisibly, because every count still adds up.
        if out["rsmi"].tolist() != list(rsmis):
            raise ValueError(
                f"the mapper returned {len(out)} reactions in a different order than it was given; "
                "its output cannot be matched to the input by position"
            )
        return list(
            zip(out["mapped_rxn"], out["confidence"].astype(float), strict=True)
        )

    return run


def mapper_tag(python: Path) -> str:
    """``rxnmapper-<version>`` of the RXNMapper environment; names its cache folder."""
    code = "import importlib.metadata as m; print(m.version('rxnmapper'))"
    result = subprocess.run(
        [str(python), "-c", code], check=True, capture_output=True, text=True
    )
    return f"rxnmapper-{result.stdout.strip()}"


def cache_dirs(root: Path, tag: str) -> tuple[Path, Path]:
    """Open and private cache folders; reactions only licensed sources have go to the private one."""
    return root / "cache" / tag, root / "private" / "cache" / tag
