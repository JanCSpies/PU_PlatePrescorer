# PU_PlatePrescorer

Pre-scoring HTE plates from literature reaction data via Positive-Unlabeled learning.
Glorius group, University of Münster — Jan Christopher Spies, Luka Praprotnik Čurič.

## Status: rebuild in progress
The first, AI-assisted prototype is archived as tag `v0-prototype`. Its results are
unverified. This repository rebuilds the pipeline in small, reviewed, tested steps;
the plan is being written in `docs/design/rebuild.md`.

To read or run the prototype without touching this checkout:

```bash
git worktree add ../PU_v0 v0-prototype
```

## Setup

```bash
uv sync
uv run pytest
```

`uv sync` installs the package and the dev tools (pytest, ruff, mypy).

## Layout
| Path | What |
|---|---|
| `src/platepu/` | the new package (grows one stage at a time) |
| `tests/` | tests for every module |

Rows are added here as new folders appear.

## Licence
Code: MIT (see LICENSE). Datasets carry their own terms; Pistachio-derived data is
never committed.
