#!/usr/bin/env python3
"""
Plot weighted precision-recall and ROC curves from pair_scores.tsv files.

Usage:
    python scripts/plot_benchmark_curves.py \\
        --pair-scores benchmark_results_rosetta_phase2_final/pair_scores.tsv \\
        --output-dir figures/

    python scripts/plot_benchmark_curves.py \\
        --pair-scores a/pair_scores.tsv b/pair_scores.tsv \\
        --labels "Run A" "Run B" --output-dir figures/
"""

import argparse
import csv
import os
import random


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]


def compute_weighted_pr(pos_scores, neg_scores, wp=0.01, seed=42):
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


def downsample(xs, ys, n=2000):
    step = max(1, len(xs) // n)
    return [xs[i] for i in range(0, len(xs), step)], \
           [ys[i] for i in range(0, len(ys), step)]


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)

    inputs = args.pair_scores
    labels = args.labels if args.labels else [os.path.basename(os.path.dirname(p))
                                               for p in inputs]
    if len(labels) < len(inputs):
        labels += [os.path.basename(os.path.dirname(p))
                   for p in inputs[len(labels):]]

    baseline_pr = args.wp / (args.wp + 1)

    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    ax_pr.axhline(baseline_pr, color="grey", linewidth=1.0, linestyle="--",
                  label=f"Random baseline ({baseline_pr:.4f})", zorder=1)

    max_prec = baseline_pr
    for path, label, color in zip(inputs, labels, COLORS):
        print(f"Loading {path}...")
        pos, neg = load_scores(path)
        print(f"  Positives: {len(pos)}, Negatives: {len(neg)}")
        precs, recs, aucpr = compute_weighted_pr(pos, neg, wp=args.wp)
        r_plot, p_plot = downsample(recs, precs)
        max_prec = max(max_prec, max(p_plot))
        ax_pr.plot(r_plot, p_plot, color=color, linewidth=1.5,
                   label=f"{label}  (AUCPR = {aucpr:.4f}, {aucpr/baseline_pr:.1f}×baseline)",
                   zorder=2)
        print(f"  AUCPR: {aucpr:.4f}  ({aucpr/baseline_pr:.2f}× baseline)")

    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel(f"Weighted Precision ($w_p = {args.wp}$)", fontsize=11)
    ax_pr.set_title("Weighted Precision–Recall Curve\n(Rosetta Stone Paired-Domain Transfer)",
                    fontsize=11)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, min(max_prec * 1.25, 1.0))
    ax_pr.legend(fontsize=8, loc="upper right")
    ax_pr.grid(True, alpha=0.3, linewidth=0.5)
    ax_pr.spines["top"].set_visible(False)
    ax_pr.spines["right"].set_visible(False)

    pr_path = os.path.join(args.output_dir, "pr_curve.pdf")
    fig_pr.savefig(pr_path, bbox_inches="tight", dpi=300)
    fig_pr.savefig(pr_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig_pr)
    print(f"Saved: {pr_path}")

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot([0, 1], [0, 1], color="grey", linewidth=1.0, linestyle="--",
                label="Random baseline (AUC = 0.5000)", zorder=1)

    for path, label, color in zip(inputs, labels, COLORS):
        pos, neg = load_scores(path)
        fprs, tprs, roc_auc = compute_roc(pos, neg)
        f_plot, t_plot = downsample(fprs, tprs)
        ax_roc.plot(f_plot, t_plot, color=color, linewidth=1.5,
                    label=f"{label}  (AUC = {roc_auc:.4f})", zorder=2)
        print(f"  ROC-AUC: {roc_auc:.4f}  ({label})")

    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("ROC Curve\n(Rosetta Stone Paired-Domain Transfer)", fontsize=11)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(True, alpha=0.3, linewidth=0.5)
    ax_roc.spines["top"].set_visible(False)
    ax_roc.spines["right"].set_visible(False)

    roc_path = os.path.join(args.output_dir, "roc_curve.pdf")
    fig_roc.savefig(roc_path, bbox_inches="tight", dpi=300)
    fig_roc.savefig(roc_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig_roc)
    print(f"Saved: {roc_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-scores", nargs="+", required=True, dest="pair_scores",
                   help="One or more pair_scores.tsv files (plotted on same axes)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Legend labels for each input (same order as --pair-scores)")
    p.add_argument("--output-dir", default="figures/", dest="output_dir")
    p.add_argument("--wp", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
