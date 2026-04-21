#!/usr/bin/env python3
"""
plot_benchmark_curves.py

Generates two figures from benchmark pair_scores.tsv:
  1. Weighted precision-recall curve (wp=0.01) with random baseline
  2. ROC curve with random baseline diagonal

Usage:
    python scripts/plot_benchmark_curves.py \
        --pair-scores benchmark_results_zhang_only/pair_scores.tsv \
        --output-dir figures/
"""

import argparse
import csv
import os
import random

def compute_weighted_pr(pos_scores, neg_scores, wp=0.01, seed=42):
    """Matches benchmark_ppi_v2.py exactly: shuffle before sort to break ties randomly."""
    P = len(pos_scores)
    all_pairs = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    precisions, recalls = [], []
    tp = fp = 0
    for score, label in all_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append((tp * wp) / (tp * wp + fp))
        recalls.append(tp / P)
    aucpr = sum(
        0.5 * (precisions[i] + precisions[i-1]) * (recalls[i] - recalls[i-1])
        for i in range(1, len(recalls))
    )
    return precisions, recalls, aucpr


def compute_roc(pos_scores, neg_scores):
    """Group tied scores together so TPR and FPR jump simultaneously — matches Mann-Whitney U."""
    from collections import defaultdict
    N_pos = len(pos_scores)
    N_neg = len(neg_scores)
    counts = defaultdict(lambda: [0, 0])
    for s in pos_scores:
        counts[s][0] += 1
    for s in neg_scores:
        counts[s][1] += 1
    fprs, tprs = [0.0], [0.0]
    tp = fp = 0
    for s in sorted(counts, reverse=True):
        tp += counts[s][0]
        fp += counts[s][1]
        fprs.append(fp / N_neg)
        tprs.append(tp / N_pos)
    roc_auc = sum(
        0.5 * (tprs[i] + tprs[i-1]) * (fprs[i] - fprs[i-1])
        for i in range(1, len(fprs))
    )
    return fprs, tprs, roc_auc


def load_scores(pair_scores_path):
    pos_scores, neg_scores = [], []
    with open(pair_scores_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            score = float(row["score"])
            if row["label"] == "positive":
                pos_scores.append(score)
            else:
                neg_scores.append(score)
    return pos_scores, neg_scores


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading scores from {args.pair_scores}...")
    pos_scores, neg_scores = load_scores(args.pair_scores)
    print(f"  Positives: {len(pos_scores)}, Negatives: {len(neg_scores)}")

    # --- Weighted PR curve ---
    precs, recs, aucpr = compute_weighted_pr(pos_scores, neg_scores, wp=args.wp)
    baseline_pr = args.wp / (args.wp + 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Downsample to 2000 points for clean rendering
    step = max(1, len(precs) // 2000)
    r_plot = [recs[i] for i in range(0, len(recs), step)]
    p_plot = [precs[i] for i in range(0, len(precs), step)]

    ax.plot(r_plot, p_plot, color="#1f77b4", linewidth=1.5,
            label=f"Structural homology (AUCPR = {aucpr:.4f})")
    ax.axhline(baseline_pr, color="grey", linewidth=1.0, linestyle="--",
               label=f"Random baseline ({baseline_pr:.4f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Weighted Precision ($w_p = 0.01$)", fontsize=11)
    ax.set_title("Weighted Precision–Recall Curve", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(p_plot) * 1.15)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pr_path = os.path.join(args.output_dir, "pr_curve.pdf")
    fig.savefig(pr_path, bbox_inches="tight", dpi=300)
    fig.savefig(pr_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved PR curve -> {pr_path}")

    # --- ROC curve ---
    fprs, tprs, roc_auc = compute_roc(pos_scores, neg_scores)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    step = max(1, len(fprs) // 2000)
    f_plot = [fprs[i] for i in range(0, len(fprs), step)]
    t_plot = [tprs[i] for i in range(0, len(tprs), step)]

    ax.plot(f_plot, t_plot, color="#d62728", linewidth=1.5,
            label=f"Structural homology (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linewidth=1.0, linestyle="--",
            label="Random baseline (AUC = 0.5000)")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    roc_path = os.path.join(args.output_dir, "roc_curve.pdf")
    fig.savefig(roc_path, bbox_inches="tight", dpi=300)
    fig.savefig(roc_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved ROC curve -> {roc_path}")

    print(f"\nAUCPR: {aucpr:.4f}  (baseline: {baseline_pr:.4f}, lift: {aucpr/baseline_pr:.2f}x)")
    print(f"ROC-AUC: {roc_auc:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-scores",
                   default="benchmark_results_zhang_only/pair_scores.tsv",
                   dest="pair_scores")
    p.add_argument("--output-dir", default="figures/", dest="output_dir")
    p.add_argument("--wp", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
