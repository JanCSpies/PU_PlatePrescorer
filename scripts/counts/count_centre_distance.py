"""What a fingerprint sees as the sphere around the reacting atom grows.

Descriptive only: no label, neither axis. Compression failed to interpolate between molecule
identity and shared chemistry -- a frequency floor leaves 21,189 distinct fingerprints at any useful
threshold, and folding holds 20,473 at width 32 and then collapses to 66 at width 8. Locality is the
remaining lever, so this measures it before anything is pre-registered.

``d`` is the number of bonds from the reacting atom within which an atom may contribute its own
radius-2 environments. d = 0 is today's ``morgan_centre``; d = None is today's ``morgan``.

Usage:
    uv run python scripts/counts/count_centre_distance.py ~/University/Phd/Dataset/Datasets
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from platepu.classify.engine import load_class
from platepu.classify.groups import reacting_group

# RDKit ships no stub for this; the parse warnings would bury the table otherwise.
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
if len(sys.argv) != 2:
    raise SystemExit("usage: scripts/counts/count_centre_distance.py <Datasets root>")
ROOT = Path(sys.argv[1]).expanduser() / "derived"
RADIUS, N_BITS = 2, 2048
DISTANCES = (0, 1, 2, 3, 4, 6, None)

rule = load_class(Path("configs/classes/amide_coupling.yml"))
rows = pd.read_parquet(
    f"{ROOT}/private/features/amide_coupling/substrates.parquet",
    columns=["key_stereo", "role", "smiles"],
)
generator = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
print(f"substrates: {len(rows):,}")

prepared = []
for smiles, role in zip(rows["smiles"], rows["role"], strict=True):
    mol = Chem.MolFromSmiles(smiles)
    _, centre = reacting_group(mol, role, rule)
    prepared.append((mol, centre, Chem.GetDistanceMatrix(mol)[centre]))

print(
    f"\n{'d':>5} {'mean roots':>11} {'mean set/mol':>13} {'distinct fps':>13} {'share unique':>13}"
)
for d in DISTANCES:
    bits, roots_used = [], []
    for mol, centre, from_centre in prepared:
        roots = (
            None if d is None else [int(i) for i in np.flatnonzero(from_centre <= d)]
        )
        counts = (
            generator.GetCountFingerprintAsNumPy(mol)
            if roots is None
            else generator.GetCountFingerprintAsNumPy(mol, fromAtoms=roots)
        )
        bits.append(counts > 0)
        roots_used.append(mol.GetNumAtoms() if roots is None else len(roots))
    present = np.stack(bits)
    distinct = len({row.tobytes() for row in present})
    label = "all" if d is None else str(d)
    print(
        f"{label:>5} {np.mean(roots_used):>11.1f} {present.sum(1).mean():>13.1f} "
        f"{distinct:>13,} {distinct / len(present):>12.1%}"
    )
