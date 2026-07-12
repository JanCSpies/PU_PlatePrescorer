"""Render the reaction x condition group catalogue as a self-contained, sortable HTML page.

Reads ``reaction_condition_groups.parquet`` (stage 3) and writes a single local HTML file with a
filter box and click-to-sort columns, so a chemist can browse the candidate plates and pick which
(reaction, condition regime) to run first. The file is written under the gitignored ``private/``
tree and is **not** published as a shareable artifact (it summarises licensed Pistachio data).

Usage:
    uv run python -m scripts_JSP.rxngroup_report --top 500
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

IN_PARQUET = Path("datasets/processed/private/rxn_groups/reaction_condition_groups_all.parquet")
OUT_HTML = Path("datasets/processed/private/rxn_groups/groups_report_all.html")

_COLUMNS = [
    ("class_label", "reaction"),
    ("metal", "metal"),
    ("cluster_ligand", "ligand"),
    ("base_family", "base"),
    ("solvent_family", "solvent"),
    ("temp_bin", "temp"),
    ("n_unique_pairs", "#pairs"),
    ("n_pairs_public", "#pairs_pub"),
    ("n_electrophiles", "#elec"),
    ("n_nucleophiles", "#nuc"),
    ("n_reports", "#reports"),
    ("yield_median", "yld_med"),
    ("yield_frac_hit", "yld_hit"),
    ("sources", "sources"),
]

_PAGE = """<title>Reaction x condition groups</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:1rem;color:#111}}
 h1{{font-size:1.2rem}} .meta{{color:#555;margin-bottom:.6rem;font-size:.85rem}}
 input{{padding:.4rem;width:22rem;margin-bottom:.6rem}}
 table{{border-collapse:collapse;font-size:.8rem;width:100%}}
 th,td{{border:1px solid #ddd;padding:.25rem .4rem;text-align:left;white-space:nowrap}}
 th{{background:#f2f2f2;cursor:pointer;position:sticky;top:0}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums}}
 tr:nth-child(even){{background:#fafafa}}
</style>
<h1>Reaction &times; condition groups &mdash; candidate plates</h1>
<div class="meta">{meta}</div>
<input id="q" placeholder="filter (e.g. Suzuki, Pd, carbonate)..." oninput="flt()">
<table id="t"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>
<script>
const NUM=new Set({numcols});
function flt(){{const q=document.getElementById('q').value.toLowerCase();
 for(const r of document.querySelectorAll('#t tbody tr'))
  r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';}}
document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{{
 const tb=document.querySelector('#t tbody'),rows=[...tb.rows],
  asc=th.dataset.asc=th.dataset.asc==='1'?'0':'1';
 rows.sort((a,b)=>{{let x=a.cells[i].textContent,y=b.cells[i].textContent;
  if(NUM.has(i)){{x=parseFloat(x)||-1e18;y=parseFloat(y)||-1e18;return asc==='1'?x-y:y-x;}}
  return asc==='1'?x.localeCompare(y):y.localeCompare(x);}});
 rows.forEach(r=>tb.appendChild(r));}});
</script>
"""


def _fmt(col: str, value: object) -> str:
    """Format one cell value for display."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if col in {"yield_median", "yield_frac_hit"}:
        return f"{float(value):.2f}"
    if col == "cluster_ligand":
        return "" if value == "na" else str(value)[:14]
    return str(value)


def main() -> None:
    """Build the sortable HTML catalogue from the group parquet."""
    parser = argparse.ArgumentParser(description="Render the group catalogue as sortable HTML.")
    parser.add_argument("--in-parquet", type=Path, default=IN_PARQUET)
    parser.add_argument("--out", type=Path, default=OUT_HTML)
    parser.add_argument("--top", type=int, default=1000, help="Rows to include (by rank).")
    args = parser.parse_args()

    df = pd.read_parquet(args.in_parquet).head(args.top)
    num_idx = [i for i, (c, _) in enumerate(_COLUMNS) if c.startswith(("n_", "yield_"))]
    head = "".join(f"<th data-asc='0'>{lbl}</th>" for _, lbl in _COLUMNS)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(
            f"<td class='{'num' if col.startswith(('n_', 'yield_')) else ''}'>"
            f"{html.escape(_fmt(col, row[col]))}</td>"
            for col, _ in _COLUMNS
        )
        body_rows.append(f"<tr>{cells}</tr>")
    total = len(pd.read_parquet(args.in_parquet))
    meta = (
        f"{total:,} groups total; showing top {len(df):,} by unique substrate pairs. "
        "Click a header to sort; type to filter."
    )
    page = _PAGE.format(
        meta=html.escape(meta), head=head, body="".join(body_rows), numcols=json.dumps(num_idx)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"Wrote {args.out}  ({len(df):,} of {total:,} groups)")


if __name__ == "__main__":
    main()
