#!/usr/bin/env python
"""
Analyze vocab entropy/min-entropy/top-k metrics on object-token steps.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stage1_analyze_head_contrib import binary_auc, write_csv
from stage1_common import torch_load_compat, write_records_csv


DEFAULT_METRICS = [
    "entropy",
    "normalized_entropy",
    "effective_vocab_size",
    "min_entropy",
    "top1_prob",
    "actual_prob",
    "actual_surprisal",
    "actual_rank",
    "top5_mass",
    "top5_entropy",
    "top10_mass",
    "top10_entropy",
    "top50_mass",
    "top50_entropy",
    "top100_mass",
    "top100_entropy",
    "bottom5_mass",
    "bottom5_entropy",
    "bottom10_mass",
    "bottom10_entropy",
    "bottom50_mass",
    "bottom50_entropy",
    "bottom100_mass",
    "bottom100_entropy",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze vocab entropy object-token metrics.")
    parser.add_argument("--entropy-file", type=str, default="stage1_vocab_entropy/stage1_vocab_entropy.pt")
    parser.add_argument("--output-dir", type=str, default="stage1_vocab_entropy")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=927)
    return parser.parse_args()


def as_float(value):
    if value == "" or value is None:
        return None
    return float(value)


def bootstrap_auc_ci(labels, scores, iters, rng):
    auc = binary_auc(labels, scores)
    if auc is None:
        return {"auc": None, "ci95": None, "n_boot": 0}
    boot = []
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    for _ in range(iters):
        idx = rng.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        cur = binary_auc(labels[idx], scores[idx])
        if cur is not None:
            boot.append(cur)
    if not boot:
        return {"auc": auc, "ci95": None, "n_boot": 0}
    boot = np.array(boot, dtype=np.float64)
    return {
        "auc": float(auc),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n_boot": int(len(boot)),
    }


def summarize_metric(records, metric, bootstrap_iters, rng):
    rows = [r for r in records if as_float(r.get(metric)) is not None]
    if not rows:
        return None
    labels = np.array([1 if int(r["label"]) == 0 else 0 for r in rows], dtype=np.int64)
    scores = np.array([float(r[metric]) for r in rows], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return None

    grounded = scores[labels == 0]
    hallucinated = scores[labels == 1]
    pooled = math.sqrt((grounded.std() ** 2 + hallucinated.std() ** 2) / 2.0)
    auc = bootstrap_auc_ci(labels, scores, bootstrap_iters, rng)
    return {
        "metric": metric,
        "n": int(len(rows)),
        "grounded_n": int(len(grounded)),
        "hallucinated_n": int(len(hallucinated)),
        "grounded_mean": float(grounded.mean()),
        "hallucinated_mean": float(hallucinated.mean()),
        "h_minus_g": float(hallucinated.mean() - grounded.mean()),
        "cohens_d_h_minus_g": float((hallucinated.mean() - grounded.mean()) / pooled) if pooled > 0 else None,
        "auc_hallucinated_high": auc["auc"],
        "auc_hallucinated_high_ci95": auc["ci95"],
        "oriented_auc": max(auc["auc"], 1.0 - auc["auc"]) if auc["auc"] is not None else None,
        "n_boot": auc["n_boot"],
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch_load_compat(args.entropy_file, map_location="cpu")
    records = payload["object_records"]

    summary = {}
    for metric in args.metrics:
        cur = summarize_metric(records, metric, args.bootstrap_iters, rng)
        if cur is not None:
            summary[metric] = cur

    ranked = sorted(
        summary.values(),
        key=lambda item: item["oriented_auc"] if item["oriented_auc"] is not None else 0.0,
        reverse=True,
    )

    with open(output_dir / "stage1_vocab_entropy_analysis_summary.json", "w") as f:
        json.dump({
            "entropy_file": args.entropy_file,
            "num_object_records": len(records),
            "num_grounded": int(sum(int(r["label"]) == 1 for r in records)),
            "num_hallucinated": int(sum(int(r["label"]) == 0 for r in records)),
            "metric_summary": summary,
            "top_metrics_by_oriented_auc": ranked[:10],
        }, f, indent=2)

    write_records_csv(records, output_dir / "stage1_vocab_entropy_object_records_flat.csv", exclude_keys={"caption"})
    write_csv(ranked, output_dir / "stage1_vocab_entropy_metric_summary.csv")

    print(json.dumps({
        "num_object_records": len(records),
        "num_grounded": int(sum(int(r["label"]) == 1 for r in records)),
        "num_hallucinated": int(sum(int(r["label"]) == 0 for r in records)),
        "top_metrics_by_oriented_auc": ranked[:8],
    }, indent=2))


if __name__ == "__main__":
    main()
