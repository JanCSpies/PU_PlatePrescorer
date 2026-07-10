"""Render the Chemlex amide plate-prescoring evaluation to a multi-page PDF.

Reads ``multiseed_metrics.json`` (per-seed + mean/std) and ``eval_arrays.npz`` (seed-averaged
production raw-ECFP test/val scores) written by ``run_multiseed_eval.py``.

Operating points are **rank-based (run the top-K of the plate)**, not an absolute-probability
threshold: plate pre-scoring deploys as a *ranking* (you run the top-K wells your budget allows),
and an absolute threshold does not transfer across substrate regions anyway. The report shows:

1. executive summary framed around plate pre-scoring (run fewer, hit more);
2. multi-seed ROC-AUC / AUPRC (transductive vs inductive, both featurizers) + bootstrap CI;
3. **confusion matrices** at a top-50%-of-plate budget (transductive vs inductive) plus a table of
   operating points (top 25 / 35 / 50 %) -- hit rate, coverage, failures skipped, TP/FP/TN/FN;
4. ROC + precision-recall curves;
5. the **plate operating curve** -- hit rate & coverage vs fraction of plate run.

Run with:  uv run python -m scripts_JSP.make_report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ART = Path("artifacts/dataset_switch")
METRICS = ART / "multiseed_metrics.json"
ARRAYS = ART / "eval_arrays.npz"
OUT_PDF = ART / "plate_prescoring_report.pdf"

BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"
GREEN, RED = "#2f855a", "#c53030"
HEADLINE_BUDGET = 0.50  # fraction of plate run for the headline confusion matrices
BUDGETS = (0.25, 0.35, 0.50)  # operating points tabulated for the deployment (inductive) regime


def _bootstrap_auc(y: np.ndarray, scores: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """Percentile 95% CI for ROC-AUC by resampling the test set (seeded for reproducibility)."""
    rng = np.random.default_rng(0)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], scores[s]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _confusion_topk(y: np.ndarray, scores: np.ndarray, frac: float) -> dict[str, float]:
    """Confusion counts + operating metrics when the top ``frac`` of the plate (by score) is run."""
    n = len(y)
    k = round(frac * n)
    order = np.argsort(-scores)
    pred = np.zeros(n, dtype=int)
    pred[order[:k]] = 1
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),  # plate hit rate
        "recall": tp / (tp + fn) if tp + fn else float("nan"),  # feasible caught
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),  # failures skipped
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "frac_run": k / n,
    }


def _operating_curve(
    scores: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frac_run, cumulative hit-rate/precision, cumulative recall) sorted by score desc."""
    order = np.argsort(-scores)
    ys = y[order]
    k = np.arange(1, len(ys) + 1)
    hits = np.cumsum(ys)
    return k / len(ys), hits / k, hits / ys.sum()


def _load() -> dict[str, dict]:
    """Load the seed-averaged test scores and derive per-regime evaluation blocks."""
    arr = np.load(ARRAYS)
    reg: dict[str, dict] = {}
    for regime in ("transductive", "inductive"):
        y = arr[f"{regime}_test_y"]
        s = arr[f"{regime}_test_scores"]
        reg[regime] = {
            "y": y,
            "s": s,
            "auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
            "ci": _bootstrap_auc(y, s),
            "base": float(y.mean()),
            "cm": _confusion_topk(y, s, HEADLINE_BUDGET),
        }
    return reg


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Render a text-only page (title + left-aligned body)."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold", color=BLUE)
    fig.text(0.08, 0.915, "─" * 78, fontsize=9, color=GREY)
    y = 0.88
    for ln in lines:
        weight = "bold" if ln.startswith("§") else "normal"
        size = 12 if ln.startswith("§") else 10.0
        fig.text(0.08, y, ln.lstrip("§"), fontsize=size, weight=weight, va="top")
        y -= 0.026 if ln.startswith("§") else (0.020 if ln else 0.012)
    pdf.savefig(fig)
    plt.close(fig)


def _draw_confusion(ax: plt.Axes, cm: dict[str, float], title: str) -> None:
    """Draw a 2x2 confusion heatmap (rows = actual, cols = run/skip) with counts + rates."""
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    total = mat.sum()
    ax.imshow(mat / total, cmap="Blues", vmin=0, vmax=0.6)
    labels = [["TN", "FP"], ["FN", "TP"]]
    colors = [[GREEN, RED], [RED, GREEN]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.15, labels[i][j], ha="center", va="center",
                    fontsize=11, weight="bold", color=colors[i][j])
            ax.text(j, i + 0.08, f"{int(mat[i, j])}", ha="center", va="center", fontsize=15)
            ax.text(j, i + 0.28, f"{mat[i, j] / total:.1%}", ha="center", va="center",
                    fontsize=9, color=GREY)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SKIP well", "RUN well"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["actual FAIL", "actual WORK"])
    ax.set_title(title, fontsize=11.5, weight="bold")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)


def main() -> None:
    """Assemble the PDF report from the saved multi-seed metrics + scores."""
    meta = json.loads(METRICS.read_text())
    metrics, seeds = meta["metrics"], meta["seeds"]
    reg = _load()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        _summary_page(pdf, reg, metrics, seeds)
        _confusion_page(pdf, reg)
        _curves_page(pdf, reg)
        _plate_page(pdf, reg)
    print(f"Wrote {OUT_PDF}")


def _summary_page(pdf: PdfPages, reg: dict, metrics: dict, seeds: list) -> None:
    """Executive summary + multi-seed metric table."""
    t, i = reg["transductive"], reg["inductive"]
    c50 = i["cm"]
    lines = [
        "§Purpose",
        "Pre-score a substrate plate (amide C-N coupling) so chemists run the wells likely to",
        "WORK and skip those likely to FAIL - more products per plate, fewer wasted experiments.",
        "Model: literature-only Positive-Unlabeled learning (258,645 amide positives mined from",
        "ORD+Pistachio; U = the candidate acid x amine matrix), one L2-regularised HistGB on",
        "substrate ECFP. Test = Chemlex wet-lab set (Nat Commun 2025), 2,234 held-out reactions,",
        "feasible = conversion >= 20%. Deployment = rank the plate, run the top-K (budget-based).",
        "",
        f"§Headline (mean over {len(seeds)} seeds; raw-ECFP production model)",
        f"  Transductive (U includes the plate's substrates):  ROC-AUC {t['auc']:.3f}"
        f"  [95% CI {t['ci'][0]:.3f}-{t['ci'][1]:.3f}]",
        f"  Inductive   (U has ZERO test substrates - novel):   ROC-AUC {i['auc']:.3f}"
        f"  [95% CI {i['ci'][0]:.3f}-{i['ci'][1]:.3f}]",
        "  => performance barely changes on NOVEL substrates: a genuine feasibility predictor, not",
        "     memorisation. (The JnJ/BH benchmark collapsed 0.72 -> 0.40 inductively.)",
        "",
        "§What it buys you (inductive, run top 50% of plate)",
        f"  hit rate {c50['precision']:.0%} vs {i['base']:.0%} blind "
        f"({c50['precision'] / i['base']:.1f}x)  |  catch {c50['recall']:.0%} of feasible  |  "
        f"skip {c50['specificity']:.0%} of failures",
        "",
        "§Multi-seed table  (ROC-AUC / AUPRC, mean +/- std)",
    ]
    for regime in ("transductive", "inductive"):
        for f in ("raw_ecfp", "corr08"):
            m = metrics[regime][f]["mean"]
            sd = metrics[regime][f]["std"]
            lines.append(
                f"  {regime:<12} {f:<9}  AUC {m['roc_auc']:.3f}+/-{sd['roc_auc']:.3f}"
                f"   AP {m['auprc']:.3f}+/-{sd['auprc']:.3f}"
            )
    lines += [
        "",
        "§How to read the rest",
        "Next: CONFUSION MATRICES at a top-50%-of-plate budget - how many failing reactions we",
        "correctly SKIP (TN) and how many working ones we RUN (TP) - plus a table of budgets. Then",
        "ROC/PR curves, then the PLATE OPERATING CURVE: hit rate & coverage vs how much you run.",
    ]
    _text_page(pdf, "Amide-coupling plate pre-scoring - evaluation report", lines)


def _confusion_page(pdf: PdfPages, reg: dict) -> None:
    """Confusion matrices at the headline budget + a budget-sweep table for deployment."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(f"Confusion matrices - run the top {HEADLINE_BUDGET:.0%} of the plate",
                 fontsize=14, weight="bold", color=BLUE, y=0.97)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.1, 1.1, 1.25], hspace=0.5, wspace=0.35,
                  top=0.9, bottom=0.07, left=0.12, right=0.9)
    for col, regime in enumerate(("transductive", "inductive")):
        r = reg[regime]
        _draw_confusion(fig.add_subplot(gs[0, col]), r["cm"],
                        f"{regime.capitalize()} (n={len(r['y'])})")
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ind = reg["inductive"]
    rows = [_confusion_topk(ind["y"], ind["s"], b) for b in BUDGETS]
    cell = [
        [f"{b:.0%}", f"{c['precision']:.1%}", f"{c['recall']:.1%}", f"{c['specificity']:.1%}",
         f"{c['tp']}", f"{c['fp']}", f"{c['tn']}", f"{c['fn']}"]
        for b, c in zip(BUDGETS, rows, strict=True)
    ]
    tbl = ax.table(
        cellText=cell,
        colLabels=["run", "hit rate", "coverage", "fails\nskipped", "TP", "FP", "TN", "FN"],
        loc="center", cellLoc="center", bbox=[0.02, 0.0, 0.96, 0.92],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    ax.set_title("Operating points for the deployment (inductive) model - pick a budget",
                 fontsize=11.5, weight="bold", pad=10)
    axt = fig.add_subplot(gs[2, :])
    axt.axis("off")
    c = reg["inductive"]["cm"]
    base = reg["inductive"]["base"]
    txt = (
        "Plain-language (inductive / novel substrates; run top 50% = 1,117 of 2,234 wells):\n"
        f"  - Of the wells we RUN, {c['precision']:.0%} actually work (vs {base:.0%} blind) "
        f"- a {c['precision'] / base:.1f}x hit-rate uplift.\n"
        f"  - We correctly SKIP {c['tn']} of {c['tn'] + c['fp']} failing reactions "
        f"({c['specificity']:.0%}) - the wasted experiments avoided.\n"
        f"  - We still capture {c['recall']:.0%} of all feasible reactions, missing "
        f"{c['fn']} true hits at this budget.\n"
        "  - Want purity over coverage? Run top 25% -> ~62% hit rate. It is a dial set by budget."
    )
    axt.text(0.02, 0.95, txt, fontsize=10.3, va="top", linespacing=1.7)
    pdf.savefig(fig)
    plt.close(fig)


def _curves_page(pdf: PdfPages, reg: dict) -> None:
    """ROC and precision-recall curves for both regimes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 6.5))
    for regime, color in (("transductive", BLUE), ("inductive", ORANGE)):
        r = reg[regime]
        fpr, tpr, _ = roc_curve(r["y"], r["s"])
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{regime} (AUC {r['auc']:.3f})")
        prec, rec, _ = precision_recall_curve(r["y"], r["s"])
        ax2.plot(rec, prec, color=color, lw=2, label=f"{regime} (AP {r['ap']:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax1.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    ax2.axhline(reg["transductive"]["base"], ls="--", color=GREY, lw=1, label="base rate")
    ax2.set(xlabel="recall", ylabel="precision", title="Precision-Recall")
    for ax in (ax1, ax2):
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Discrimination curves (seed-averaged production model)",
                 fontsize=14, weight="bold", color=BLUE)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


def _plate_page(pdf: PdfPages, reg: dict) -> None:
    """The plate operating curve: hit rate + coverage vs fraction of plate run."""
    fig, ax = plt.subplots(figsize=(11.69, 7.2))
    for regime, color in (("transductive", BLUE), ("inductive", ORANGE)):
        r = reg[regime]
        frac, prec, rec = _operating_curve(r["s"], r["y"])
        ax.plot(frac, prec, color=color, lw=2.2, label=f"{regime}: hit rate (precision)")
        ax.plot(frac, rec, color=color, lw=1.6, ls="--", label=f"{regime}: coverage (recall)")
    ax.axhline(reg["inductive"]["base"], color=RED, ls=":", lw=1.5,
               label=f"random hit rate ({reg['inductive']['base']:.1%})")
    ax.set_xlabel("fraction of plate run (top-scored wells first)", fontsize=11)
    ax.set_ylabel("hit rate  /  coverage", fontsize=11)
    ax.set_title("Plate operating curve - run fewer wells, hit more often",
                 fontsize=14, weight="bold", color=BLUE)
    ax.legend(fontsize=9.5, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    note = ("Read: run only the top-scored fraction X of the plate (x-axis) -> solid = fraction\n"
            "that actually work (hit rate); dashed = fraction of all feasible reactions still\n"
            "captured (coverage). The gap above the red line is the experiments-saved value.")
    fig.text(0.1, 0.02, note, fontsize=9.5, color=GREY)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
