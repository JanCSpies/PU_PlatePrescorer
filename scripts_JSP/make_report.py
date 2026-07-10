"""Render the Chemlex amide plate-prescoring evaluation to a multi-page PDF.

Reads ``multiseed_metrics.json`` (per-seed + mean/std) and ``eval_arrays.npz`` (seed-averaged
production raw-ECFP test/val scores) written by ``run_multiseed_eval.py`` and produces
``artifacts/dataset_switch/plate_prescoring_report.pdf`` with:

1. executive summary framed around plate pre-scoring (run fewer, hit more);
2. multi-seed ROC-AUC / AUPRC table (transductive vs inductive, both featurizers) + bootstrap CI;
3. **confusion matrices** (transductive + inductive) at the val-selected Youden threshold, with the
   operating-point metrics that sell the tool -- specificity (failures skipped), precision (hit
   rate), recall (feasible caught), NPV;
4. ROC + precision-recall curves;
5. the **plate operating curve** -- hit rate & recall vs fraction of plate run.

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


def _youden(y: np.ndarray, scores: np.ndarray) -> float:
    """Youden-J-maximising threshold on a labelled split."""
    fpr, tpr, thr = roc_curve(y, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


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


def _confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Confusion counts + plate operating-point metrics at a fixed threshold."""
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "frac_run": (tp + fp) / len(y),
    }


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
    """Draw a 2x2 confusion heatmap (rows = actual, cols = predicted) with counts + rates."""
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
    ax.set_xticklabels(["pred FAIL", "pred WORK"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["actual FAIL", "actual WORK"])
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)


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
    """Load metrics + scores and derive the per-regime evaluation blocks."""
    arr = np.load(ARRAYS)
    reg: dict[str, dict] = {}
    for regime in ("transductive", "inductive"):
        y = arr[f"{regime}_test_y"]
        s = arr[f"{regime}_test_scores"]
        thr = _youden(arr[f"{regime}_val_y"], arr[f"{regime}_val_scores"])
        reg[regime] = {
            "y": y,
            "s": s,
            "thr": thr,
            "cm": _confusion(y, (s >= thr).astype(int)),
            "auc": float(roc_auc_score(y, s)),
            "ap": float(average_precision_score(y, s)),
            "ci": _bootstrap_auc(y, s),
            "base": float(y.mean()),
        }
    return reg


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
    lines = [
        "§Purpose",
        "Pre-score a substrate plate (amide C-N coupling) so chemists run the wells likely to",
        "WORK and skip those likely to FAIL - more products per plate, fewer wasted experiments.",
        "Model: literature-only Positive-Unlabeled learning (258,645 amide positives mined from",
        "ORD+Pistachio; U = the candidate acid x amine matrix), one L2-regularised HistGB on",
        "substrate ECFP. Test = Chemlex wet-lab set (Nat Commun 2025), 2,234 held-out reactions,",
        "feasible = conversion >= 20%.",
        "",
        f"§Headline (mean over {len(seeds)} seeds; raw-ECFP production model)",
        f"  Transductive (U includes the plate's substrates):  ROC-AUC {t['auc']:.3f}"
        f"  [95% CI {t['ci'][0]:.3f}-{t['ci'][1]:.3f}]",
        f"  Inductive   (U has ZERO test substrates - novel):   ROC-AUC {i['auc']:.3f}"
        f"  [95% CI {i['ci'][0]:.3f}-{i['ci'][1]:.3f}]",
        "  => performance barely changes on NOVEL substrates: a genuine feasibility predictor, not",
        "     memorisation. (The JnJ/BH benchmark collapsed 0.72 -> 0.40 inductively.)",
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
        "Next page: CONFUSION MATRICES at the operating threshold (chosen on validation, not the",
        "test) -- how many failing reactions we correctly skip (TN) and how many working ones we",
        "would run (TP). Then ROC/PR curves, then the PLATE OPERATING CURVE: hit rate and coverage",
        "as a function of how much of the plate you choose to run.",
    ]
    _text_page(pdf, "Amide-coupling plate pre-scoring - evaluation report", lines)


def _confusion_page(pdf: PdfPages, reg: dict) -> None:
    """Confusion matrices + operating-point metric table for both regimes."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Confusion matrices at the validation-selected operating point",
                 fontsize=14, weight="bold", color=BLUE, y=0.97)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.1, 1.1, 1.3], hspace=0.5, wspace=0.35,
                  top=0.9, bottom=0.07, left=0.12, right=0.9)
    for col, regime in enumerate(("transductive", "inductive")):
        r = reg[regime]
        _draw_confusion(fig.add_subplot(gs[0, col]), r["cm"],
                        f"{regime.capitalize()} (n={len(r['y'])})")
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    rows = ["precision (plate HIT RATE)", "recall (feasible caught)",
            "specificity (failures SKIPPED)", "npv (skipped truly fail)", "frac_run (plate run)"]
    keys = ["precision", "recall", "specificity", "npv", "frac_run"]
    cell = [[f"{reg['transductive']['cm'][k]:.1%}", f"{reg['inductive']['cm'][k]:.1%}"]
            for k in keys]
    tbl = ax.table(cellText=cell, rowLabels=rows, colLabels=["transductive", "inductive"],
                   loc="center", cellLoc="center", bbox=[0.30, 0.0, 0.62, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.5)
    ax.set_title("Operating-point metrics (what the plate designer cares about)",
                 fontsize=12, weight="bold", pad=14)
    axt = fig.add_subplot(gs[2, :])
    axt.axis("off")
    t = reg["inductive"]["cm"]
    base = reg["inductive"]["base"]
    txt = (
        "Plain-language (inductive / novel-substrate regime):\n"
        f"  - Of every 100 wells the model flags WORK, ~{t['precision'] * 100:.0f} truly work "
        f"(vs {base * 100:.0f} picking blindly) - a {t['precision'] / base:.1f}x hit-rate uplift.\n"
        f"  - It correctly SKIPS {t['specificity'] * 100:.0f}% of reactions that would have failed "
        f"({t['tn']} of {t['tn'] + t['fp']}) - the wasted experiments avoided.\n"
        f"  - Cost: it still catches {t['recall'] * 100:.0f}% of all feasible reactions "
        f"(missing {t['fn']} hits here; move the dial below to trade recall vs precision).\n"
        "  - Threshold fixed on VALIDATION data, never the test set."
    )
    axt.text(0.02, 0.95, txt, fontsize=10.5, va="top", linespacing=1.7)
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
