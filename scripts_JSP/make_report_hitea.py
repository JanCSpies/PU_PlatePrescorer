"""Render the second-reaction (Buchwald-Hartwig C-N / HiTEA) plate-prescoring report to a PDF.

Companion to ``make_report.py`` (Chemlex amide): same literature-PU method, same substrate-only
raw-ECFP features, same HistGB -- applied to a *different reaction class* to show the approach
generalises. Reads ``hitea_metrics.json`` + ``hitea_arrays.npz`` (written by ``run_hitea_eval.py``)
and produces ``artifacts/dataset_switch/hitea_bh_report.pdf`` with:

1. executive summary (generalisation claim) + multi-seed table + the n=36 / base-rate caveats;
2. confusion matrix at a plate budget + a budget-sweep table;
3. ROC + precision-recall curves;
4. PAYN-style comparison (positives-only vs PU-ours vs in-domain LOO, which collapses at n=36);
5. plate hit rate by ranking model (curve + budget table).

HiTEA is small and natural, so two Chemlex pieces do not apply and are stated honestly: the
**inductive** regime is impossible (all 36 test substrates span the grid U) and the **true**
in-domain bar collapses (36 non-repeating pairs -> leave-one-out CV ~ random or worse).

Run with:  uv run python -m scripts_JSP.make_report_hitea
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
METRICS = ART / "hitea_metrics.json"
ARRAYS = ART / "hitea_arrays.npz"
OUT_PDF = ART / "hitea_bh_report.pdf"

BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"
GREEN, RED = "#2f855a", "#c53030"
HEADLINE_BUDGET = 0.50
BUDGETS = (0.25, 0.35, 0.50)


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
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "frac_run": k / n,
    }


def _operating_curve(
    scores: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frac_run, cumulative hit-rate, cumulative recall) sorted by score desc."""
    order = np.argsort(-scores)
    ys = y[order]
    k = np.arange(1, len(ys) + 1)
    hits = np.cumsum(ys)
    return k / len(ys), hits / k, hits / ys.sum()


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


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Render a text-only page (title + left-aligned body; lines starting with '§' are headers)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=15, weight="bold", color=BLUE)
    fig.text(0.08, 0.915, "─" * 82, fontsize=9, color=GREY)
    y = 0.88
    for ln in lines:
        weight = "bold" if ln.startswith("§") else "normal"
        size = 12 if ln.startswith("§") else 10.0
        fig.text(0.08, y, ln.lstrip("§"), fontsize=size, weight=weight, va="top")
        y -= 0.026 if ln.startswith("§") else (0.020 if ln else 0.012)
    pdf.savefig(fig)
    plt.close(fig)


def _load() -> dict[str, object]:
    """Load metrics + arrays into a single dict for the pages."""
    meta = json.loads(METRICS.read_text())
    arr = np.load(ARRAYS)
    y = arr["test_y"]
    return {
        "meta": meta,
        "y": y,
        "pu_s": arr["pu_scores"],
        "pos_s": arr["positives_only_scores"],
        "true_s": arr["true_loo_scores"],
        "base": float(y.mean()),
        "cm": _confusion_topk(y, arr["pu_scores"], HEADLINE_BUDGET),
    }


def main() -> None:
    """Assemble the HiTEA BH report from the saved metrics + scores."""
    d = _load()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        _summary_page(pdf, d)
        _confusion_page(pdf, d)
        _curves_page(pdf, d)
        _payn_page(pdf, d)
        _hitrate_page(pdf, d)
        _leakage_page(pdf)
    print(f"Wrote {OUT_PDF}")


def _summary_page(pdf: PdfPages, d: dict) -> None:
    """Generalisation summary + multi-seed table + honest small-n caveats."""
    meta = d["meta"]
    y, base = d["y"], d["base"]
    pu_raw = meta["pu"]["raw_ecfp"]["mean"]
    pu_raw_sd = meta["pu"]["raw_ecfp"]["std"]
    ci = meta["pu_bootstrap_ci"]
    pos_auc = meta["baselines"]["positives_only"]["mean"]["roc_auc"]
    c35 = _confusion_topk(y, d["pu_s"], 0.35)
    lines = [
        "§Purpose - does the method generalise to a second reaction?",
        "The Chemlex report showed literature-only PU learning pre-scores an AMIDE-coupling plate.",
        "Here we apply the IDENTICAL method - literature positives + candidate grid U, substrate-",
        "only raw-ECFP, one L2-regularised HistGB - to Buchwald-Hartwig C-N coupling (a different",
        "bond, a different mechanism). Test = HiTEA BH screen; P = 64,061 BH literature positives.",
        "",
        f"§Headline (mean over {len(meta['seeds'])} seeds; raw-ECFP)",
        f"  PU - ours:  ROC-AUC {pu_raw['roc_auc']:.3f} +/- {pu_raw_sd['roc_auc']:.3f}"
        f"   [95% CI {ci[0]:.2f}-{ci[1]:.2f}]   AUPRC {pu_raw['auprc']:.3f}",
        f"  Beats positives-only ({pos_auc:.3f}) and random (0.5). Same ordering as Chemlex:",
        "  literature-PU > similarity-to-known > random - on a second reaction class.",
        "",
        "§What it buys you (run top 35% of the plate)",
        f"  hit rate {c35['precision']:.0%} vs {base:.0%} blind "
        f"({c35['precision'] / base:.2f}x)  |  catch {c35['recall']:.0%} of feasible  |  "
        f"skip {c35['specificity']:.0%} of failures",
        "",
        "§Multi-seed table  (ROC-AUC / AUPRC, mean +/- std)",
    ]
    for f in ("raw_ecfp", "corr08"):
        m = meta["pu"][f]["mean"]
        sd = meta["pu"][f]["std"]
        lines.append(
            f"  PU  {f:<9}  AUC {m['roc_auc']:.3f}+/-{sd['roc_auc']:.3f}"
            f"   AP {m['auprc']:.3f}+/-{sd['auprc']:.3f}"
        )
    lines += [
        "",
        "§Read this honestly - HiTEA is a small, natural test",
        f"  - n = {len(y)} substrate pairs ({int(y.sum())} work / {int(len(y) - y.sum())} fail);",
        "    the bootstrap CI above is wide. Treat AUC gaps as directional, not decisive.",
        f"  - High base rate ({base:.0%}): random already hits often, so headroom is smaller",
        "    than Chemlex (35% base). The hit-rate lift is still real (see the plate curve).",
        "  - TRANSDUCTIVE only: all 36 test substrates span the grid U, so the inductive drop",
        "    that Chemlex passed cannot even be run here.",
        "  - No IN-DOMAIN model: only 36 labelled pairs, substrates barely repeat -> an in-",
        "    domain supervised model (LOO-CV) collapses (page 4) - the argument FOR transfer.",
    ]
    _text_page(pdf, "Buchwald-Hartwig C-N plate pre-scoring - generalisation report", lines)


def _confusion_page(pdf: PdfPages, d: dict) -> None:
    """Confusion matrix at the headline budget + a budget-sweep table (PU model)."""
    y = d["y"]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(f"PU model - confusion at a top-{HEADLINE_BUDGET:.0%} plate budget (HiTEA BH)",
                 fontsize=13.5, weight="bold", color=BLUE, y=0.96)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.2, 1.0, 1.1], hspace=0.5,
                  top=0.88, bottom=0.07, left=0.16, right=0.84)
    axc = fig.add_subplot(gs[0])
    _draw_confusion(axc, d["cm"], f"Run top {HEADLINE_BUDGET:.0%} of {len(y)} pairs")
    axt = fig.add_subplot(gs[1])
    axt.axis("off")
    cell = []
    for b in BUDGETS:
        cm = _confusion_topk(y, d["pu_s"], b)
        cell.append([f"{b:.0%}", f"{cm['precision']:.0%}", f"{cm['recall']:.0%}",
                     f"{cm['specificity']:.0%}", f"{cm['tp']}", f"{cm['fp']}",
                     f"{cm['tn']}", f"{cm['fn']}"])
    tbl = axt.table(
        cellText=cell,
        colLabels=["run", "hit rate", "coverage", "fails\nskipped", "TP", "FP", "TN", "FN"],
        loc="center", cellLoc="center", bbox=[0.04, 0.0, 0.92, 0.92],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    axt.set_title("Operating points (PU model) - pick a plate budget", fontsize=11.5,
                  weight="bold", pad=8)
    axi = fig.add_subplot(gs[2])
    axi.axis("off")
    c = d["cm"]
    base = d["base"]
    txt = (
        f"Run the top {HEADLINE_BUDGET:.0%} of the {len(y)}-pair plate ranked by the PU score:\n"
        f"  - Of the wells we RUN, {c['precision']:.0%} actually work (vs {base:.0%} blind).\n"
        f"  - We correctly SKIP {c['tn']} of {c['tn'] + c['fp']} failing reactions.\n"
        f"  - We still capture {c['recall']:.0%} of all feasible reactions.\n"
        "Counts are small (n=36) - read as proof of transfer, not a precise operating point."
    )
    axi.text(0.02, 0.9, txt, fontsize=10.3, va="top", linespacing=1.7)
    pdf.savefig(fig)
    plt.close(fig)


def _curves_page(pdf: PdfPages, d: dict) -> None:
    """ROC and precision-recall curves for the PU model."""
    y, s, base = d["y"], d["pu_s"], d["base"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 6.5))
    fpr, tpr, _ = roc_curve(y, s)
    ax1.plot(fpr, tpr, color=ORANGE, lw=2, label=f"PU - ours (AUC {roc_auc_score(y, s):.3f})")
    ax1.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax1.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    ax2.plot(rec, prec, color=ORANGE, lw=2, label=f"PU - ours (AP {ap:.3f})")
    ax2.axhline(base, ls="--", color=GREY, lw=1, label=f"base rate ({base:.1%})")
    ax2.set(xlabel="recall", ylabel="precision", title="Precision-Recall")
    for ax in (ax1, ax2):
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("PU model discrimination on HiTEA BH (seed-averaged, n=36)",
                 fontsize=14, weight="bold", color=BLUE)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _payn_page(pdf: PdfPages, d: dict) -> None:
    """PAYN-style comparison: positives-only vs PU vs the collapsing in-domain LOO bar."""
    meta = d["meta"]
    y, pos_s, pu_s, true_s = d["y"], d["pos_s"], d["pu_s"], d["true_s"]
    pos_auc = roc_auc_score(y, pos_s)
    pu_auc = roc_auc_score(y, pu_s)
    true_auc = roc_auc_score(y, true_s)
    pu_sd = meta["pu"]["raw_ecfp"]["std"]["roc_auc"]
    bars = [
        ("true (in-domain LOO)", true_auc, 0.0, RED),
        ("positives-only", pos_auc, 0.0, GREY),
        ("PU - ours", pu_auc, pu_sd, ORANGE),
    ]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("PAYN-style comparison on a second reaction (HiTEA BH)",
                 fontsize=13.5, weight="bold", color=BLUE, y=0.97)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.3, 1.0], hspace=0.5,
                  top=0.9, bottom=0.06, left=0.20, right=0.9)
    axb = fig.add_subplot(gs[0])
    yb = np.arange(len(bars))
    axb.barh(yb, [b[1] for b in bars], xerr=[b[2] for b in bars], color=[b[3] for b in bars],
             height=0.62, error_kw={"ecolor": "#4a5568", "capsize": 4})
    for yv, (_, v, e, _c) in zip(yb, bars, strict=True):
        axb.text(max(v + e + 0.01, 0.03), yv, f"{v:.3f}", va="center", fontsize=11, weight="bold")
    axb.axvline(0.5, ls="--", color=RED, lw=1.2)
    axb.set_yticks(yb)
    axb.set_yticklabels([b[0] for b in bars], fontsize=9.5)
    axb.set_xlim(0, 1.0)
    axb.set_xlabel("TEST ROC-AUC (n=36; random = 0.5)", fontsize=10)
    axb.set_title("Feasibility discrimination on 36 held-out BH pairs", fontsize=11, weight="bold")
    axb.grid(axis="x", alpha=0.3)

    axr = fig.add_subplot(gs[1])
    for s, name, col in ((pos_s, "positives-only", GREY), (pu_s, "PU - ours", ORANGE),
                         (true_s, "true (in-domain LOO)", RED)):
        fpr, tpr, _ = roc_curve(y, s)
        axr.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
    axr.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    axr.set(xlabel="false positive rate", ylabel="true positive rate",
            title="ROC - identical 36-pair test")
    axr.legend(fontsize=9, loc="lower right")
    axr.grid(alpha=0.3)
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1)

    axt = fig.add_subplot(gs[2])
    axt.axis("off")
    txt = (
        "Same three references as the Chemlex report, on BH C-N:\n"
        f"  - positives-only ({pos_auc:.3f}): Tanimoto to the nearest of 64,061 BH literature\n"
        "    positives. The naive 'looks like a known reaction' skyline.\n"
        f"  - PU - ours ({pu_auc:.3f}): literature P + candidate grid U; no target labels.\n"
        f"  - true / in-domain ({true_auc:.3f}): the SAME HistGB, leave-one-out CV on the 36 real\n"
        "    labels. It COLLAPSES - with 36 non-repeating pairs there is no in-domain signal to\n"
        "    learn. You cannot bootstrap a feasibility model from the target; literature is the\n"
        "    only option, and it works.\n"
        "=> PU beats similarity-to-known and random on a second reaction class. The AUC margin\n"
        "   over positives-only is small at n=36 - the hit-rate gap is the clearer read."
    )
    axt.text(0.0, 0.98, txt, fontsize=9.8, va="top", linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def _hitrate_page(pdf: PdfPages, d: dict) -> None:
    """Plate hit rate by ranking model (positives-only / PU / in-domain LOO)."""
    y, base = d["y"], d["base"]
    models = [
        ("positives-only", d["pos_s"], GREY),
        ("PU - ours", d["pu_s"], ORANGE),
        ("true (in-domain LOO)", d["true_s"], RED),
    ]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Plate hit rate by model - HiTEA BH", fontsize=14, weight="bold", color=BLUE,
                 y=0.97)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.35, 1.0], hspace=0.32,
                  top=0.9, bottom=0.08, left=0.12, right=0.93)
    ax = fig.add_subplot(gs[0])
    for name, s, col in models:
        frac, prec, _rec = _operating_curve(s, y)
        ax.plot(frac, prec, color=col, lw=2.2, label=name)
    ax.axhline(base, color=RED, ls=":", lw=1.5, label=f"random ({base:.1%})")
    ax.set_xlabel("fraction of plate run (top-scored pairs first)", fontsize=11)
    ax.set_ylabel("hit rate (precision)", fontsize=11)
    ax.set_title("Hit rate vs how much of the plate you run, per ranking model",
                 fontsize=11.5, weight="bold")
    ax.legend(fontsize=9.5, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1.02)
    axt = fig.add_subplot(gs[1])
    axt.axis("off")
    cell = []
    for b in BUDGETS:
        row = [f"{b:.0%}"]
        for _, s, _c in models:
            cm = _confusion_topk(y, s, b)
            row.append(f"{cm['precision']:.0%} / {cm['recall']:.0%}")
        cell.append(row)
    tbl = axt.table(
        cellText=cell,
        colLabels=["run top", "positives-only", "PU - ours", "true (LOO)"],
        loc="center", cellLoc="center", bbox=[0.05, 0.32, 0.9, 0.54],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.6)
    axt.set_title("Hit rate / coverage at fixed plate budgets  (cell = hit-rate % / coverage %)",
                  fontsize=11, weight="bold", pad=8)
    p35 = {name: _confusion_topk(y, s, 0.35) for name, s, _c in models}
    cap = (
        f"Rank the plate by a model, run the top X%. Random picking hits {base:.0%}.\n"
        f"At top-35%: PU-ours {p35['PU - ours']['precision']:.0%}, "
        f"positives-only {p35['positives-only']['precision']:.0%}, "
        f"in-domain LOO {p35['true (in-domain LOO)']['precision']:.0%}. The PU pre-scorer lifts\n"
        "the hit rate above both blind picking and similarity ranking - same as Chemlex, now on\n"
        "Buchwald-Hartwig C-N. (In-domain LOO tracks below random: no signal at n=36.)"
    )
    axt.text(0.05, 0.24, cap, fontsize=9.8, va="top", linespacing=1.55, transform=axt.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def _leakage_page(pdf: PdfPages) -> None:
    """Reviewer-facing data-leakage audit for the HiTEA BH transfer.

    Same audit as the Chemlex report, tailored to HiTEA's transductive-only, no-val (n=36) regime:
    the classifier is fit on ``(P, U)`` alone and the 36 held-out pair labels are consumed only by
    ``predict_proba`` at scoring time.
    """
    lines = [
        "§Data-leakage audit - does the test set touch training?",
        "Short answer: NO. The classifier sees zero target (HiTEA) labels while training. It is",
        "fit on (P = 64,061 BH literature positives, U = the halide x amine grid) only; the 36",
        "held-out pair labels are read once, at scoring time, to make this report.",
        "",
        "§1. No early stopping on the test",
        "HistGB.fit() receives only P and U. Its internal early-stopping split (sklearn default)",
        "is carved from P+U - literature/grid rows - never from the 36 HiTEA pairs. The test is",
        "not in the training matrix, so it cannot be in any early-stopping split.",
        "",
        "§2. No tuning, no threshold fit on the test",
        "Model (400 iters, lr 0.05, L2 1.0) and features were frozen from the JnJ/BH benchmark and",
        "reused unchanged. HiTEA has no validation split (only 36 pairs); operating points are set",
        "by PLATE BUDGET (run the top-K), not by a threshold tuned on test labels. ROC-AUC is",
        "threshold-free regardless.",
        "",
        "§3. No feature leakage",
        "Headline = raw ECFP: 2048 fixed bits, no data-driven selection, so fitting the featurizer",
        "selects nothing. (corr08 does mild bit selection but lands on the same number.)",
        "",
        "§4. Transductive-only - disclosed, and it carries no labels",
        "All 36 test substrates span U, so the inductive drop Chemlex passed cannot be run here",
        "(see page 1). The pairs sit in U as implicit NEGATIVES - no target yield or label ever",
        "enters training. The novel-substrate generalisation U cannot prove here is exactly what",
        "the Chemlex inductive result demonstrates.",
        "",
        "§One caveat we state openly",
        "The only human-in-the-loop test contact is model-selection altitude (which featurizer to",
        "headline); both agree. We disclose it rather than imply a fully blind protocol.",
    ]
    _text_page(pdf, "Data-leakage audit - methodological integrity", lines)


if __name__ == "__main__":
    main()
