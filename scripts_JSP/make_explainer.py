"""Render a plain-language explainer PDF for the literature-PU plate-prescoring project.

This is a *teaching* document, not a results report: it explains, for a reader who is **not** a
Positive-Unlabeled (PU) learning expert, every concept behind the pipeline -- what P and U are and
how they were built, how the test/validation splits work, transductive vs inductive evaluation, the
reporting-bias (SCAR vs SAR) problem and the shadow features that fix it, every model in the PU arm
(HistGB+L2, the naive baseline, nnPU, iwPU, Elkan-Noto, Bagging-PU, Spy, prop_ips, SAR-EM), and the
two ranking metrics (ROC-AUC and AUPRC) with the reason AUPRC matters most for plate design.

Figures are drawn from the *real* project artifacts where possible (the shadow P-vs-U separability
histograms from ``explainer_cache.npz``; the ROC/PR curves from ``eval_arrays*.npz``; leaderboard
bars from the per-model metric JSONs), so the pictures match the numbers in the lab journal.

Run with:  uv run python -m scripts_JSP.make_explainer
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ART = Path("artifacts/dataset_switch")
CACHE = ART / "explainer_cache.npz"
EVAL_CHAMP = ART / "eval_arrays.npz"
EVAL_PROP = ART / "eval_arrays_propips.npz"
CHAMPION = ART / "multiseed_metrics.json"
ALLMODELS = ART / "allmodels"
OUT_PDF = ART / "PU_plate_prescoring_explainer.pdf"

BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"
GREEN, RED, INK = "#2f855a", "#c53030", "#1a202c"
TEAL, PURPLE = "#2c7a7b", "#6b46c1"

#: Pretty display names for the leaderboard bar chart (registry label -> label shown to the reader).
MODEL_NAMES = {
    "sup_histgb_l2 (champion)": "HistGB+L2  (naive P-vs-U champion)",
    "prop_ips": "prop_ips  (beyond-SCAR)",
    "sar_em": "SAR-EM  (beyond-SCAR)",
    "elkan": "Elkan-Noto",
    "bagging": "Bagging-PU",
    "spy": "Spy (2-step)",
    "iw_pu": "iwPU",
    "nnpu": "nnPU",
}


# --------------------------------------------------------------------------------------------------
# Auto-paginating text renderer (so long sections never overflow a page).
# --------------------------------------------------------------------------------------------------
def _wrap(blocks: list[str]) -> list[tuple[str, str]]:
    """Turn semantic blocks into wrapped (kind, text) render lines.

    Block syntax: ``"§Heading"`` -> a bold heading; ``""`` -> a vertical gap; ``"- text"`` -> a
    bullet (hanging-indented on wrap); anything else -> a normal paragraph (word-wrapped).
    """
    out: list[tuple[str, str]] = []
    for b in blocks:
        if b == "":
            out.append(("gap", ""))
        elif b.startswith("§"):
            out.append(("head", b[1:]))
        elif b.startswith("- "):
            segs = textwrap.wrap(b[2:], 92)
            for i, seg in enumerate(segs):
                out.append(("bullet" if i == 0 else "cont", seg))
        else:
            for seg in textwrap.wrap(b, 98):
                out.append(("body", seg))
    return out


def _flow(pdf: PdfPages, title: str, blocks: list[str]) -> None:
    """Render blocks across as many A4 pages as needed with a running header."""
    lines = _wrap(blocks)
    i, page = 0, 0
    while i < len(lines):
        fig = plt.figure(figsize=(8.27, 11.69))
        hdr = title if page == 0 else f"{title}   (continued)"
        fig.text(0.07, 0.955, hdr, fontsize=15, weight="bold", color=BLUE, va="top")
        fig.text(0.07, 0.936, "─" * 92, fontsize=8, color=GREY, va="top")
        y = 0.905
        while i < len(lines) and y > 0.06:
            kind, txt = lines[i]
            if kind == "gap":
                y -= 0.013
            elif kind == "head":
                y -= 0.010
                fig.text(0.07, y, txt, fontsize=11.5, weight="bold", color=INK, va="top")
                y -= 0.030
            elif kind == "bullet":
                fig.text(0.075, y, "•", fontsize=10, color=BLUE, va="top")
                fig.text(0.098, y, txt, fontsize=9.6, va="top")
                y -= 0.0198
            elif kind == "cont":
                fig.text(0.098, y, txt, fontsize=9.6, va="top")
                y -= 0.0198
            else:
                fig.text(0.07, y, txt, fontsize=9.6, va="top")
                y -= 0.0198
            i += 1
        pdf.savefig(fig)
        plt.close(fig)
        page += 1


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    face: str,
    edge: str = "none",
    fs: float = 9.5,
    tc: str = "white",
) -> tuple[float, float]:
    """Draw a rounded box with centred text; return its centre (for arrow anchoring)."""
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=1.2,
            facecolor=face,
            edgecolor=edge,
            mutation_aspect=1,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        weight="bold",
        wrap=True,
    )
    return x + w / 2, y + h / 2


def _arrow(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], color: str = INK) -> None:
    """Draw a fat arrow from point ``a`` to point ``b``."""
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color=color,
            shrinkA=6,
            shrinkB=6,
        )
    )


# --------------------------------------------------------------------------------------------------
# Figure pages.
# --------------------------------------------------------------------------------------------------
def _page_pipeline(pdf: PdfPages) -> None:
    """One-glance schematic of the pipeline (literature -> P, grid -> U, train, score plate)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.07, 0.955, "The pipeline at a glance", fontsize=15, weight="bold", color=BLUE, va="top"
    )
    fig.text(0.07, 0.936, "─" * 92, fontsize=8, color=GREY, va="top")
    ax = fig.add_axes((0.05, 0.40, 0.9, 0.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _box(
        ax,
        (0.3, 7.4),
        (4.2, 1.6),
        "P  =  POSITIVES\n258,645 amide couplings that\nWORKED (mined from the literature:\nORD + "
        "Pistachio)",
        BLUE,
        fs=9,
    )
    _box(
        ax,
        (5.5, 7.4),
        (4.2, 1.6),
        "U  =  UNLABELED\n62,832 candidate acid x amine\ncombinations. Feasibility unknown\n(used "
        "as stand-in 'negatives')",
        TEAL,
        fs=9,
    )
    _box(
        ax,
        (2.6, 4.4),
        (4.8, 1.5),
        "CLASSIFIER  learns to tell\nP apart from U\n(HistGB on molecular fingerprints)",
        PURPLE,
        fs=9.5,
    )
    _box(
        ax,
        (2.6, 1.7),
        (4.8, 1.4),
        "FEASIBILITY SCORE  for every well\n->  rank the plate, run the top-K",
        ORANGE,
        fs=9.5,
    )
    _arrow(ax, (2.4, 7.4), (3.8, 5.9), BLUE)
    _arrow(ax, (7.6, 7.4), (6.2, 5.9), TEAL)
    _arrow(ax, (5.0, 4.4), (5.0, 3.1), PURPLE)

    lines = [
        "Read top-to-bottom. We never have a clean list of reactions that FAIL (chemists rarely",
        "publish failures), so instead of positives-vs-negatives we do Positive-Unlabeled "
        "learning:",
        "the model learns what separates known-good reactions (P) from the sea of untried",
        "candidates (U), and turns that into a 0-1 feasibility score used to design better plates.",
    ]
    fig.text(0.07, 0.33, "\n".join(lines), fontsize=10, va="top", color=INK, linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def _page_trans_ind(pdf: PdfPages) -> None:
    """Two mini-grids: test substrates present in U (transductive) vs removed (inductive)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.07,
        0.955,
        "Transductive vs inductive  (did U contain the test substrates?)",
        fontsize=14,
        weight="bold",
        color=BLUE,
        va="top",
    )
    fig.text(0.07, 0.936, "─" * 92, fontsize=8, color=GREY, va="top")

    n = 6
    test_cells = {(1, 4), (3, 1), (4, 4), (2, 2)}  # cells that belong to the held-out test plate
    for col, (regime, keep) in enumerate((("Transductive", True), ("Inductive", False))):
        ax = fig.add_axes((0.10 + col * 0.46, 0.56, 0.34, 0.30))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(regime, fontsize=12, weight="bold", color=INK, pad=8)
        for r in range(n):
            for c in range(n):
                is_test = (r, c) in test_cells
                if is_test and not keep:
                    face, ec = "white", RED  # removed from U
                elif is_test:
                    face, ec = ORANGE, ORANGE  # test cell, present in U
                else:
                    face, ec = "#cbd5e0", "#cbd5e0"  # ordinary candidate in U
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.45, r - 0.45),
                        0.9,
                        0.9,
                        facecolor=face,
                        edgecolor=ec,
                        linewidth=1.4,
                        hatch="///" if (is_test and not keep) else None,
                    )
                )
        ax.set_xlabel("amines  ->", fontsize=8, color=GREY)
        ax.set_ylabel("acids  ->", fontsize=8, color=GREY)
    fig.text(
        0.11,
        0.52,
        "grey = ordinary U candidate      orange = test-plate well (in U)      "
        "red hatched = test well REMOVED from U",
        fontsize=8.5,
        color=GREY,
    )

    blocks = [
        "§The one question",
        "The classifier trains on the grid U. The only difference between the two setups is "
        "whether "
        "the substrates from the held-out test plate are sitting inside U during training (as "
        "unlabeled points, never with their answers).",
        "",
        "§Transductive (left)  ->  our score 0.795",
        "U is built from ALL Chemlex substrates, so the test wells (orange) are physically in the "
        "training grid. The model never sees their react/no-react labels, but it has 'met' those "
        "exact molecules as feature vectors. Analogy: you were handed the exam questions to "
        "practise "
        "on beforehand -- without the answer key.",
        "",
        "§Inductive (right)  ->  our score 0.784",
        "The test substrates are deleted from U (red, hatched). The model trains having never "
        "encountered them in any form, then must score them cold. Analogy: you study the textbook, "
        ""
        "then face brand-new questions on exam day.",
        "",
        "§Why it matters",
        "Transductive can flatter a model: it might quietly memorise the shape of the test "
        "substrates even without labels. Inductive is the honest deployment test -- a genuinely "
        "new "
        "plate. Our numbers barely move (0.795 -> 0.784), so the model is a real feasibility "
        "predictor, not a memoriser. (For contrast, the older JnJ/BH benchmark collapsed 0.72 -> "
        "0.40 going inductive: that WAS memorisation being exposed.)",
    ]
    _flow_after(fig, blocks, y0=0.44)
    pdf.savefig(fig)
    plt.close(fig)


def _flow_after(fig: plt.Figure, blocks: list[str], y0: float) -> None:
    """Render wrapped text into an existing figure starting at height ``y0`` (single page)."""
    y = y0
    for kind, txt in _wrap(blocks):
        if kind == "gap":
            y -= 0.011
        elif kind == "head":
            y -= 0.006
            fig.text(0.07, y, txt, fontsize=11, weight="bold", color=INK, va="top")
            y -= 0.026
        elif kind in ("bullet", "cont"):
            if kind == "bullet":
                fig.text(0.075, y, "•", fontsize=10, color=BLUE, va="top")
            fig.text(0.098, y, txt, fontsize=9.4, va="top")
            y -= 0.0188
        else:
            fig.text(0.07, y, txt, fontsize=9.4, va="top")
            y -= 0.0188


def _page_reporting_bias(pdf: PdfPages) -> None:
    """Schematic: which working reactions get reported under SCAR (fair coin) vs SAR (bias)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.07,
        0.955,
        "Reporting bias:  SCAR (wishful) vs SAR (reality)",
        fontsize=14,
        weight="bold",
        color=BLUE,
        va="top",
    )
    fig.text(0.07, 0.936, "─" * 92, fontsize=8, color=GREY, va="top")

    rng = np.random.default_rng(3)
    pop = rng.uniform(0, 1, 60)  # popularity/availability of each working reaction's substrates
    for row, (name, reported) in enumerate(
        (
            (
                "SCAR  (assumed):  reporting is a fair coin -- independent of popularity",
                rng.random(60) < 0.4,
            ),
            (
                "SAR  (real):  popular/available substrates are far more likely to be reported",
                rng.random(60) < (0.05 + 0.9 * pop),
            ),
        )
    ):
        ax = fig.add_axes((0.10, 0.70 - row * 0.16, 0.82, 0.11))
        ax.scatter(
            pop[~reported],
            np.zeros((~reported).sum()) + 0.5,
            s=42,
            facecolor="white",
            edgecolor=GREY,
            linewidth=1.1,
            label="stays hidden in U",
            zorder=2,
        )
        ax.scatter(
            pop[reported],
            np.zeros(reported.sum()) + 0.5,
            s=48,
            color=BLUE,
            label="gets reported (-> P)",
            zorder=3,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["obscure", "popular / cheap / in-stock"], fontsize=8.5)
        ax.set_title(name, fontsize=9.5, weight="bold", color=INK, loc="left")
        if row == 0:
            ax.legend(
                fontsize=8, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.9)
            )
    fig.text(
        0.10,
        0.55,
        "Every dot is a reaction that genuinely WORKS. Filled = published (lands "
        "in P). Hollow = never written up (stays in U).",
        fontsize=8.5,
        color=GREY,
    )

    blocks = [
        "§The trap",
        "The naive assumption (SCAR = Selected Completely At Random) is that any working reaction "
        "has the SAME chance of being published. It doesn't. Chemists preferentially report "
        "reactions built from popular, cheap, commercially available building blocks (SAR = "
        "Selected At Random given features). So P is a biased sample of what works: over-full of "
        "famous substrates, under-full of obscure ones.",
        "",
        "§Why that hurts",
        "If you just train 'P vs U', the model partly learns what makes a substrate POPULAR "
        "instead "
        "of what makes a reaction WORK -- because popular substrates flood P. The fix (next page) "
        "is "
        "to model the reporting bias explicitly using features about popularity/availability that "
        "say nothing about the chemistry itself.",
    ]
    _flow_after(fig, blocks, y0=0.50)
    pdf.savefig(fig)
    plt.close(fig)


def _page_separability(pdf: PdfPages) -> None:
    """Real shadow-propensity score distributions: amide (well separated) vs BH (overlapping)."""
    c = np.load(CACHE)
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.07,
        0.955,
        "Shadow features & 'P-vs-U separability'  (a thermometer for bias)",
        fontsize=13.5,
        weight="bold",
        color=BLUE,
        va="top",
    )
    fig.text(0.07, 0.936, "─" * 92, fontsize=8, color=GREY, va="top")

    panels = (
        (
            "Amide  (Chemlex)",
            c["amide_p"],
            c["amide_u"],
            float(c["amide_auc"]),
            "STRONG bias -> prop_ips WINS (+0.024)",
        ),
        (
            "Buchwald-Hartwig  (HiTEA)",
            c["bh_p"],
            c["bh_u"],
            float(c["bh_auc"]),
            "WEAK bias -> prop_ips ties/loses",
        ),
    )
    for col, (name, ps, us, auc, verdict) in enumerate(panels):
        ax = fig.add_axes((0.10 + col * 0.46, 0.60, 0.36, 0.26))
        bins = np.linspace(0, 1, 26)
        ax.hist(us, bins=bins, density=True, color=TEAL, alpha=0.6, label="U (candidates)")
        ax.hist(ps, bins=bins, density=True, color=BLUE, alpha=0.6, label="P (reported)")
        ax.set_title(f"{name}\nseparability AUC = {auc:.3f}", fontsize=10, weight="bold")
        ax.set_xlabel("shadow 'reportedness' score", fontsize=8.5)
        ax.set_yticks([])
        ax.legend(fontsize=8, loc="upper center")
        ax.text(
            0.5,
            -0.30,
            verdict,
            transform=ax.transAxes,
            ha="center",
            fontsize=8.6,
            weight="bold",
            color=(GREEN if col == 0 else RED),
        )

    blocks = [
        "§What the shadow features are",
        "Two deliberately NON-chemical numbers per building block: (1) how often it appears in a "
        "huge reaction corpus (report frequency, ~832k reactions) and (2) whether it is "
        "commercially in-stock (ZINC). They capture fame/availability, NOT whether the chemistry "
        "works. prop_ips uses them to model 'how likely was this reported?' and reweights the "
        "positives to undo the bias -- up-weighting positives that look like the obscure, "
        "under-reported corner of U, down-weighting the over-reported popular ones.",
        "",
        "§What 'separability' measures",
        "Take ONLY those shadow features and ask: can they tell reported positives (P) apart from "
        "the candidate grid (U)? The answer is an AUC (see the metrics page). Read it as: 'how "
        "much "
        "reporting bias is actually there to correct?'",
        "",
        "§Why it explains Chemlex-yes / HiTEA-no",
        "- Amide = 0.931: the two humps barely overlap. Reported amides really are much more "
        "'popular' than the general grid -- a strong, real bias. prop_ips has genuine signal to "
        "exploit, and it beats the champion by +0.024.",
        "- BH = 0.755: the humps overlap a lot. Reported BH substrates look almost like the grid, "
        "so there is little identifiable bias. prop_ips's reweighting is near-uniform noise -- it "
        "mostly adds variance and does NOT help (-0.033 on the production featurizer).",
        "So separability is a cheap up-front check: high -> expect prop_ips to help; low -> don't. "
        ""
        "The beyond-SCAR win is conditional on there being an identifiable bias, not a free lunch.",
    ]
    _flow_after(fig, blocks, y0=0.50)
    pdf.savefig(fig)
    plt.close(fig)


def _page_taxonomy(pdf: PdfPages) -> None:
    """Box diagram grouping PU models into SCAR wrappers vs beyond-SCAR, all on a HistGB base."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.07,
        0.955,
        "The models in the PU arm  (a family tree)",
        fontsize=14,
        weight="bold",
        color=BLUE,
        va="top",
    )
    fig.text(0.07, 0.936, "─" * 92, fontsize=8, color=GREY, va="top")
    ax = fig.add_axes((0.05, 0.45, 0.9, 0.45))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _box(ax, (3.2, 8.6), (3.6, 1.1), "PU methods\n(only P + U, no true negatives)", INK, fs=10)
    _box(
        ax,
        (0.3, 5.8),
        (4.2, 1.4),
        "SCAR family\n'reporting is a fair coin'\n(assume constant labeling rate)",
        BLUE,
        fs=9,
    )
    _box(
        ax,
        (5.5, 5.8),
        (4.2, 1.4),
        "Beyond-SCAR family\nmodel the reporting bias\nusing shadow features",
        ORANGE,
        fs=9,
    )
    _arrow(ax, (5.0, 8.6), (2.4, 7.2), BLUE)
    _arrow(ax, (5.0, 8.6), (7.6, 7.2), ORANGE)

    scar_kids = [
        "HistGB+L2 (naive P-vs-U) = champion",
        "Elkan-Noto",
        "Bagging-PU",
        "Spy (2-step)",
        "nnPU",
        "iwPU",
    ]
    # A single short connector from each family box down to its column of children (no clutter).
    _arrow(ax, (2.4, 5.8), (2.4, 5.36), "#a0aec0")
    _arrow(ax, (7.6, 5.8), (7.6, 5.36), "#dd9a5c")
    for k, name in enumerate(scar_kids):
        yb = 4.7 - k * 0.72
        _box(ax, (0.3, yb), (4.2, 0.56), name, "#e2e8f0", edge=BLUE, fs=8.5, tc=INK)
    for k, name in enumerate(["prop_ips  (IPS reweighting)", "SAR-EM  (joint EM)"]):
        yb = 4.7 - k * 0.72
        _box(ax, (5.5, yb), (4.2, 0.56), name, "#feebc8", edge=ORANGE, fs=8.5, tc=INK)

    fig.text(
        0.07,
        0.40,
        "Everything shares the SAME engine: a HistGB classifier (see next page).\n"
        "The families differ only in how they treat U's hidden positives and the reporting bias.",
        fontsize=9.2,
        color=INK,
        va="top",
        linespacing=1.5,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _leaderboard() -> list[tuple[str, float]]:
    """Load transductive/raw_ecfp mean ROC-AUC for every model, sorted best-first."""
    rows: list[tuple[str, float]] = []
    champ = json.loads(CHAMPION.read_text())["metrics"]["transductive"]["raw_ecfp"]["mean"][
        "roc_auc"
    ]
    rows.append(("sup_histgb_l2 (champion)", float(champ)))
    for path in sorted(ALLMODELS.glob("*.json")):
        payload = json.loads(path.read_text())
        auc = payload["metrics"]["transductive"]["raw_ecfp"]["mean"]["roc_auc"]
        rows.append((payload["model_label"], float(auc)))
    return sorted(rows, key=lambda kv: kv[1], reverse=True)


def _page_leaderboard(pdf: PdfPages) -> None:
    """Horizontal bar chart of every PU model's TEST ROC-AUC, prop_ips highlighted."""
    rows = _leaderboard()
    champ = next(v for k, v in rows if "champion" in k)
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.07,
        0.955,
        "Leaderboard  (Chemlex amide, TEST ROC-AUC, 5 seeds)",
        fontsize=14,
        weight="bold",
        color=BLUE,
        va="top",
    )
    fig.text(0.07, 0.936, "─" * 92, fontsize=8, color=GREY, va="top")
    ax = fig.add_axes((0.34, 0.50, 0.58, 0.38))
    names = [MODEL_NAMES.get(k, k) for k, _ in rows]
    vals = [v for _, v in rows]
    colors = [
        ORANGE if k == "prop_ips" else (BLUE if "champion" in k else "#a0aec0") for k, _ in rows
    ]
    ypos = np.arange(len(rows))[::-1]
    ax.barh(ypos, vals, color=colors, height=0.66)
    for yv, v in zip(ypos, vals, strict=True):
        ax.text(v + 0.004, yv, f"{v:.3f}", va="center", fontsize=9, weight="bold")
    ax.axvline(champ, ls="--", color=BLUE, lw=1.2)
    ax.axvline(0.5, ls=":", color=RED, lw=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8.6)
    ax.set_xlim(0.45, 0.86)
    ax.set_xlabel("TEST ROC-AUC  (0.5 = random, dashed blue = champion)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    blocks = [
        "§How to read it",
        "Higher = better at ranking feasible reactions above infeasible ones on the held-out "
        "plate. "
        "The dashed blue line is the naive HistGB champion; red dots at 0.5 would be coin-flips.",
        "",
        "§The story in one look",
        "- prop_ips (orange) is the only method that clearly beats the champion (+0.024). It is "
        "the "
        "beyond-SCAR reweighting from the separability page.",
        "- Elkan-Noto and Bagging-PU essentially tie the champion (+0.008 / +0.004) -- because "
        "they "
        "ARE HistGB underneath; the PU wrapper adds little.",
        "- Spy, nnPU and iwPU sit clearly below. nnPU and iwPU collapsed to 0.644 (identical): "
        "their "
        "reweighting degenerated to 'do nothing' in fingerprint space (details on the model "
        "pages).",
        "- SAR-EM (beyond-SCAR too) underperforms here: same shadow inputs as prop_ips, but its "
        "iterative relabelling is high-variance and hurts the classifier. Mechanism matters, not "
        "just having the shadow features.",
    ]
    _flow_after(fig, blocks, y0=0.44)
    pdf.savefig(fig)
    plt.close(fig)


def _page_roc_pr(pdf: PdfPages) -> None:
    """Real ROC and PR curves from the production model, with the imbalance intuition."""
    arr = np.load(EVAL_PROP if EVAL_PROP.exists() else EVAL_CHAMP)
    y, s = arr["transductive_test_y"], arr["transductive_test_scores"]
    base = float(y.mean())
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.07,
        0.955,
        "The two ranking metrics:  ROC-AUC and AUPRC",
        fontsize=14,
        weight="bold",
        color=BLUE,
        va="top",
    )
    fig.text(0.07, 0.936, "─" * 92, fontsize=8, color=GREY, va="top")

    ax1 = fig.add_axes((0.10, 0.70, 0.36, 0.19))
    fpr, tpr, _ = roc_curve(y, s)
    ax1.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {roc_auc_score(y, s):.3f}")
    ax1.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax1.set(xlabel="false-positive rate", ylabel="true-positive rate (recall)", title="ROC curve")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_axes((0.58, 0.70, 0.36, 0.19))
    prec, rec, _ = precision_recall_curve(y, s)
    ax2.plot(
        rec, prec, color=ORANGE, lw=2, label=f"AP (AUPRC) = {average_precision_score(y, s):.3f}"
    )
    ax2.axhline(base, ls=":", color=RED, lw=1.3, label=f"random = {base:.2f}")
    ax2.set(
        xlabel="recall (coverage)", ylabel="precision (hit rate)", title="Precision-Recall curve"
    )
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3)

    blocks = [
        "Both grade a RANKING -- how well the model sorts truly-feasible reactions to the top. "
        "Neither needs you to pick a cutoff. Both run 0 to 1. Curves above are the real production "
        ""
        "model on the Chemlex test.",
        "",
        "§ROC-AUC  (left)",
        "Plain meaning: the chance a random working reaction scores higher than a random failing "
        "one. 0.5 = coin flip, 1.0 = perfect. It treats both classes symmetrically and is "
        "insensitive to imbalance -- a big chunk of the score comes from correctly dumping the "
        "many "
        "obvious failures at the bottom. That can look flattering even if the TOP of your list is "
        "polluted, because easy true-negatives pad the number.",
        "",
        "§AUPRC  (right, = Average Precision)",
        "Plots precision (of the wells we'd RUN, how many actually work = plate hit rate) against "
        "recall (of all working reactions, how many we caught). It ignores true-negatives and "
        "focuses on the positive class. Its 'random' baseline is not 0.5 but the base rate (dotted "
        ""
        "red). It speaks directly to: 'when I grab the top-scored wells, how many are real?'",
        "",
        "§Why AUPRC matters most for plate pre-scoring",
        "Deployment = rank the wells, run only the top-K your budget allows. You care about the "
        "PURITY at the top of the list -- exactly what AUPRC rewards. Failures dominate here (only "
        ""
        f"{base:.0%} of the test wells work), and ROC-AUC gets easy credit for shoving those "
        "failures down -- not what you are paid for. AUPRC throws that freebie away and reports "
        "the "
        "hit rate you will actually get.",
        "",
        "§A number example",
        "1,000 candidate wells, 50 truly work (5%), 950 fail; you run the top 50. A model can post "
        ""
        "ROC-AUC ~0.90 yet put only 15 real hits in its top 50 -> a 30% hit rate. AUPRC would read "
        ""
        "~0.3, honestly reflecting that the top of the list is only 30% pure. Two models with the "
        "same ROC-AUC can differ a lot in AUPRC; the higher-AUPRC one packs more real hits into "
        "the "
        "top -> more molecules per plate. (On amide, prop_ips lifted AUPRC 0.593 -> 0.632.)",
    ]
    _flow_after(fig, blocks, y0=0.635)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
# Text-only pages.
# --------------------------------------------------------------------------------------------------
def _page_title(pdf: PdfPages) -> None:
    """Cover page + reading guide."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(
        0.5,
        0.80,
        "Predicting reaction feasibility",
        ha="center",
        fontsize=22,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.5, 0.755, "from literature-only data", ha="center", fontsize=22, weight="bold", color=INK
    )
    fig.text(0.5, 0.70, "A plain-language explainer", ha="center", fontsize=14, color=BLUE)
    fig.text(
        0.5,
        0.66,
        "Positive-Unlabeled learning for smarter chemistry plates",
        ha="center",
        fontsize=11,
        color=GREY,
        style="italic",
    )
    fig.text(0.5, 0.60, "─" * 60, ha="center", fontsize=8, color=GREY)
    guide = [
        "What you will find inside (no PU-learning background assumed):",
        "",
        "  1.  The pipeline at a glance",
        "  2.  What Positive-Unlabeled (PU) learning is, and why we use it",
        "  3.  How the data was built:  P, U, and the test / validation splits",
        "  4.  Transductive vs inductive evaluation",
        "  5.  Reporting bias:  SCAR vs SAR",
        "  6.  Shadow features and 'P-vs-U separability'",
        "  7.  Every model in the PU arm, one at a time",
        "  8.  The leaderboard",
        "  9.  ROC-AUC and AUPRC, and why AUPRC rules for plate design",
        " 10.  Key takeaways + glossary",
        "",
        "Every figure is drawn from the project's real numbers, so the pictures match the",
        "lab journal. Terms in bold are collected in the glossary at the end.",
    ]
    fig.text(0.13, 0.52, "\n".join(guide), fontsize=10.5, va="top", color=INK, linespacing=1.55)
    fig.text(
        0.5,
        0.06,
        "Glorius group - Built with Claude: Life Sciences - branch JSP-dataset-switch",
        ha="center",
        fontsize=8.5,
        color=GREY,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _build_cache() -> None:
    """(Re)compute the shadow-propensity score distributions used by the separability figure.

    Runs the shadow-only P-vs-U logistic for both datasets and caches the (sampled) per-set score
    arrays + the in-sample separability AUC to :data:`CACHE`, so :func:`_page_separability` renders
    from a light file and the whole explainer regenerates from scratch with no extra manual step.
    """
    import pandas as pd
    from scripts_JSP.run_dataset_switch import AMIDE_P, MERGED_BH, build_chemlex, build_hitea
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from pubench.features.propensity import ShadowFeatureBuilder

    rng = np.random.default_rng(0)
    builder = ShadowFeatureBuilder()

    def _scores(e_p: np.ndarray, e_u: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        x = np.vstack([e_p, e_u])
        s = np.r_[np.ones(len(e_p)), np.zeros(len(e_u))]
        scaler = StandardScaler().fit(x)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(x), s)
        allsc = lr.predict_proba(scaler.transform(x))[:, 1]
        return allsc[: len(e_p)], allsc[len(e_p) :], float(roc_auc_score(s, allsc))

    def _sample(a: np.ndarray, n: int) -> np.ndarray:
        return a if len(a) <= n else a[rng.choice(len(a), n, replace=False)]

    cx, hi = build_chemlex(), build_hitea()
    ak = pd.read_parquet(AMIDE_P, columns=["acid_key", "amine_key"])
    ap_p, ap_u, ap_auc = _scores(
        builder.from_keys(ak["acid_key"], ak["amine_key"]),
        builder.from_smiles(cx.unlabeled.iloc[:, 0], cx.unlabeled.iloc[:, 1]),
    )
    bk = pd.read_parquet(MERGED_BH, columns=["aryl_key", "amine_key"])
    bh_p, bh_u, bh_auc = _scores(
        builder.from_keys(bk["aryl_key"], bk["amine_key"]),
        builder.from_smiles(hi.unlabeled.iloc[:, 0], hi.unlabeled.iloc[:, 1]),
    )
    np.savez(
        CACHE,
        amide_p=_sample(ap_p, 20000),
        amide_u=_sample(ap_u, 20000),
        amide_auc=ap_auc,
        bh_p=_sample(bh_p, 20000),
        bh_u=bh_u,
        bh_auc=bh_auc,
    )
    print(f"built {CACHE}  (amide sep {ap_auc:.3f}, BH sep {bh_auc:.3f})")


def main() -> None:
    """Assemble the explainer PDF (building the separability cache first if it is missing)."""
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        _build_cache()
    with PdfPages(OUT_PDF) as pdf:
        _page_title(pdf)
        _page_pipeline(pdf)
        _flow(pdf, "What is Positive-Unlabeled (PU) learning?", _PU_INTRO)
        _flow(pdf, "How the data was built:  P, U, test and validation", _DATA)
        _page_trans_ind(pdf)
        _page_reporting_bias(pdf)
        _page_separability(pdf)
        _page_taxonomy(pdf)
        _flow(pdf, "The models, one at a time", _MODELS)
        _page_leaderboard(pdf)
        _page_roc_pr(pdf)
        _flow(pdf, "Key takeaways & glossary", _GLOSSARY)
    print(f"Wrote {OUT_PDF}")


# --------------------------------------------------------------------------------------------------
# Long-form text content.
# --------------------------------------------------------------------------------------------------
_PU_INTRO = [
    "§The problem: we have successes but no clean failures",
    "To train an ordinary classifier you need positive examples (reactions that work) AND negative "
    ""
    "examples (reactions that fail). Chemistry literature gives us the first but almost never the "
    "second: papers report what worked; failed attempts are rarely published. So a labelled list "
    "of "
    "'this reaction fails' barely exists.",
    "",
    "§The PU idea: positives + an unlabeled pool",
    "Positive-Unlabeled (PU) learning is built exactly for this. We keep two sets:",
    "- P (positives): reactions known to work -- mined from the literature.",
    "- U (unlabeled): a big pool of candidate reactions whose outcome we do NOT know. Some would "
    "work, some would fail; we simply have not labelled them.",
    "The trick is to treat U as 'stand-in negatives'. Most random candidates fail, so U behaves "
    "roughly like a noisy negative set, and a classifier trained to separate P from U learns "
    "something real about feasibility -- without ever needing a curated failure list.",
    "",
    "§The catch that makes it interesting",
    "U is not a clean negative set: it secretly contains hidden positives (candidates that WOULD "
    "work, we just have not tried them). If we ignore that, we systematically punish good "
    "reactions "
    "that happen to sit in U. Most of the models in this project are different clever answers to "
    "the same question: 'U is contaminated with hidden positives and P is a biased sample -- how "
    "do "
    "we still recover the true feasibility signal?'",
    "",
    "§Why we bother: better plates",
    "A plate is a tray of many reactions run at once (high-throughput experimentation). If we can "
    "score every candidate well 0-1 for feasibility BEFORE running it, we can run the wells most "
    "likely to work and skip the likely failures -- more real molecules per plate, less wasted "
    "lab effort. That pre-scoring is the whole point of the model.",
]

_DATA = [
    "§P -- the positives  (258,645 amide couplings)",
    "We mined two big reaction databases (the Open Reaction Database and Pistachio) for amide "
    "couplings -- the reaction that joins a carboxylic ACID and an AMINE into an amide. For each "
    "reported reaction we kept just the two substrates (acid + amine) and canonicalised each "
    "molecule to a standard identity key (an 'InChIKey-14', a fingerprint of the molecule's "
    "connectivity) so the same molecule drawn two ways counts once. After de-duplication: 258,645 "
    "unique acid x amine positive pairs. In code this is the parquet loaded by build_chemlex().",
    "",
    "§U -- the unlabeled candidate grid  (62,832 pairs)",
    "We took every acid and every amine that appears in the target (Chemlex) dataset -- 272 acids "
    "and 231 amines -- and formed the full combinatorial grid: 272 x 231 = 62,832 candidate pairs. "
    ""
    "That grid IS U: every acid-amine combination one could imagine putting on a plate, most never "
    ""
    "actually run. In code: the cartesian product built by _matrix() inside build_chemlex().",
    "",
    "§TEST -- the held-out wet-lab answers  (2,234 reactions)",
    "The Chemlex dataset (a published wet-lab study) actually ran a set of amide reactions and "
    "measured the outcome. We call a reaction FEASIBLE if its measured conversion is at least 20% "
    "(that 20% is the yes/no threshold, not a rate). Chemlex ships a 'Both_Unseen' split: the test "
    ""
    "reactions use acids AND amines that never appear in training -- a hard, honest generalisation "
    ""
    "test. Test = 2,234 reactions, of which 35.5% are feasible (the 'base rate').",
    "",
    "§VALIDATION -- for choosing a cut-off only  (4,960 reactions)",
    "The other half of the labelled Chemlex data (4,960 reactions, 23.7% feasible) is the "
    "validation set. We never train on it. We use it for ONE thing: if we need to convert scores "
    "into a hard run/skip decision, we pick the threshold on validation (the point that best "
    "separates feasible from infeasible), then apply it to the test set. This keeps the test set "
    "untouched by any tuning.",
    "",
    "§The second dataset -- HiTEA (Buchwald-Hartwig)",
    "To check whether findings generalise to a DIFFERENT reaction, we repeat everything on HiTEA, "
    "a "
    "Buchwald-Hartwig C-N coupling screen. Here P = 64,061 mined BH positives, U = a 33 x 28 = 924 "
    ""
    "aryl-halide x amine grid, and TEST = just 36 substrate pairs (a pair counts as feasible if "
    "any "
    "tested condition gives >=10% yield). HiTEA is small, so there is no validation split and the "
    "36-well test gives very wide error bars -- useful as a sanity check, not a precise "
    "measurement.",
    "",
    "§Two featurizations (how molecules become numbers)",
    "A model cannot read a molecule; it reads a vector. We use ECFP fingerprints: each molecule "
    "becomes a 2,048-long bit-vector marking which chemical substructures it contains. "
    "'raw_ecfp' uses all 2,048 bits; 'corr08' is a smaller de-correlated version. raw_ecfp is our "
    "production/headline featurization.",
]

_MODELS = [
    "All of these produce a 0-1 feasibility score. They share one engine and differ only in how "
    "they cope with U being a contaminated, biased negative set.",
    "",
    "§HistGB + L2  -- the engine (and the champion)",
    "HistGB = Histogram Gradient-Boosted trees: it builds hundreds of small decision trees one "
    "after another, each fixing the previous ones' mistakes, and adds up their votes. It is fast, "
    "handles thousands of fingerprint bits well, and is the base learner inside EVERY method here. "
    ""
    "'L2' is a regularisation dial that discourages over-confident trees (guards against "
    "over-fitting). Used directly -- just train HistGB to separate P from U -- it is the 'naive "
    "P-vs-U' baseline, and it proved so strong that it is our CHAMPION (0.771).",
    "",
    "§Supervised / naive P-vs-U",
    "The simplest PU approach: label all of P as 1, all of U as 0, and train the classifier as if "
    "U "
    "were true negatives. Ignores the hidden-positive contamination entirely. Surprisingly hard "
    "to beat, because most of U really does fail. (This IS the HistGB+L2 champion.)",
    "",
    "§Elkan-Noto",
    "A classic PU correction. Under the SCAR assumption there is a single constant c = "
    "'probability "
    "a positive got labelled'. Elkan-Noto estimates c (using held-out known positives) and then "
    "rescales the naive classifier's scores by 1/c to recover true feasibility probabilities. On "
    "our data it essentially ties the champion -- because the ranking it produces is the same, "
    "only "
    "the numbers are rescaled.",
    "",
    "§Bagging-PU",
    "'Bagging' = bootstrap aggregating. It repeatedly draws a random small sample of U to act as "
    "negatives, trains a HistGB against P each time, and averages the many models. Averaging over "
    "many different negative samples reduces the noise from U's contamination. Robust; on our data "
    ""
    "it matches the champion (it is HistGB many times over).",
    "",
    "§Spy (two-step)",
    "First plant a few known positives ('spies') from P into U, train a preliminary classifier, "
    "and "
    "see what scores the spies get. Any U point scoring lower than the spies is probably a genuine "
    ""
    "negative ('reliable negative'). Then retrain properly on P vs those reliable negatives. "
    "Clever, "
    "but on our data it discarded useful signal and landed below the champion.",
    "",
    "§nnPU  (non-negative PU)",
    "A principled risk-correction method. It uses an estimate of the overall positive fraction "
    "(the "
    "'class prior') to mathematically subtract off the hidden positives inside U, with a "
    "'non-negative' guard that stops the correction from running away. Powerful in theory; here it "
    ""
    "collapsed to 0.644 -- with our prior and fingerprint space the correction effectively "
    "cancelled out, leaving little signal.",
    "",
    "§iwPU  (importance-weighted PU)",
    "Reweights points by a density ratio (how much more common a region is in U than in P) so the "
    "training distribution is corrected. On our data the density ratio collapsed to roughly "
    "uniform "
    "-- so it did nothing and gave the exact same 0.644 as nnPU. The lesson feeds directly into "
    "prop_ips: importance weighting only helps if you weight in the RIGHT space.",
    "",
    "§prop_ips  -- beyond-SCAR, the winner",
    "Same idea as iwPU (reweight the positives) but done in SHADOW space -- the popularity / "
    "availability features -- instead of the chemistry-fingerprint space where iwPU degenerated. "
    "Because reported positives really are more 'popular' than the grid (separability 0.931 on "
    "amide), the weights are meaningful: it up-weights positives that resemble the under-reported "
    "corner of U and down-weights over-reported ones, then trains the champion HistGB with those "
    "weights. Result on amide: 0.795, the best (+0.024 over the champion). But it only helps where "
    ""
    "the reporting bias is identifiable -- it did NOT help on BH (separability 0.755).",
    "",
    "§SAR-EM  -- beyond-SCAR, the ambitious cousin",
    "Instead of a one-shot reweighting, SAR-EM jointly and iteratively fits TWO things: the "
    "feasibility model f(x) on chemistry features AND the reporting propensity e(x) on shadow "
    "features, alternating in an Expectation-Maximisation loop. In principle the most complete "
    "correction; in practice its repeated relabelling of U is high-variance and, on our data, hurt "
    ""
    "the classifier (below the champion). It shows that HAVING the shadow features is not enough "
    "-- "
    "how you use them decides whether they help.",
]

_GLOSSARY = [
    "§The five things worth remembering",
    "- We predict reaction feasibility from literature-only data using Positive-Unlabeled "
    "learning: "
    "positives P (reactions that worked) vs an unlabeled candidate grid U, no failure list needed.",
    "- The point is plate pre-scoring: rank candidate wells, run the top-K, get more real "
    "molecules "
    "per plate.",
    "- Inductive (novel substrates) barely trails transductive (0.784 vs 0.795): the model "
    "generalises, it does not just memorise.",
    "- Correcting the literature's reporting bias (prop_ips, beyond-SCAR) is the only thing that "
    "clearly beat the naive HistGB champion on amide (+0.024) -- but ONLY where the bias is "
    "identifiable (separability 0.931 amide vs 0.755 BH: no win on BH).",
    "- AUPRC, not ROC-AUC, is the metric that tracks what a plate actually yields, because it "
    "measures purity at the top of the ranked list and ignores the easy failures.",
    "",
    "§Glossary",
    "- P (positives): reactions known to work, mined from the literature (258,645 amide pairs).",
    "- U (unlabeled): the candidate acid x amine grid, outcomes unknown, used as stand-in "
    "negatives "
    "(62,832 pairs).",
    "- Base rate: the fraction of a set that is actually feasible (Chemlex test = 35.5%).",
    "- Feasible: a reaction whose measured conversion clears the threshold (>=20% for Chemlex).",
    "- Transductive: test substrates are present in U during training (as unlabeled points).",
    "- Inductive: test substrates are removed from U -- the honest 'brand-new plate' test.",
    "- SCAR: the (false) assumption that any positive has a constant chance of being reported.",
    "- SAR: the reality -- reporting depends on features (popularity/availability).",
    "- Shadow features: non-chemical popularity/availability signals (report frequency, ZINC "
    "in-stock) used to model the reporting bias.",
    "- Separability: AUC of a shadow-only P-vs-U classifier -- how much reporting bias exists to "
    "correct (high -> beyond-SCAR helps).",
    "- HistGB: histogram gradient-boosted trees, the shared engine of every model here.",
    "- ROC-AUC: chance a random positive outranks a random negative; imbalance-insensitive.",
    "- AUPRC / Average Precision: area under the precision-recall curve; tracks hit rate at the "
    "top "
    "of the list; the key plate-design metric.",
    "- Precision = hit rate (of wells we run, how many work). Recall = coverage (of feasible "
    "reactions, how many we catch).",
    "- ECFP: a 2,048-bit fingerprint turning a molecule into a vector the model can read.",
]


if __name__ == "__main__":
    main()
