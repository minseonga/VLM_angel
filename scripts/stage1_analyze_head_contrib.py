#!/usr/bin/env python
"""
Analyze grounded vs hallucinated head-logit contribution distributions.

This is Stage 1 / D3:
  - compare per-record distribution summaries
  - compare per-layer/per-head actual-token contribution
  - quantify wrong-object direction where GT object token targets are available
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stage1_common import torch_load_compat


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 D3: analyze head contribution distributions.")
    parser.add_argument("--contrib-file", type=str, default="stage1_outputs/stage1_head_logit_contrib.pt")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs")
    parser.add_argument("--start-layer", type=int, default=5)
    parser.add_argument("--end-layer", type=int, default=18)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def binary_auc(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average ranks for ties.
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg_rank = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = ranks[pos].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def describe_by_label(rows, metric):
    out = {}
    for label, name in [(1, "grounded"), (0, "hallucinated")]:
        vals = np.array([float(row[metric]) for row in rows if int(row["label"]) == label], dtype=np.float64)
        if len(vals) == 0:
            out[name] = None
            continue
        out[name] = {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "q10": float(np.quantile(vals, 0.10)),
            "q50": float(np.quantile(vals, 0.50)),
            "q90": float(np.quantile(vals, 0.90)),
        }
    g = out.get("grounded")
    h = out.get("hallucinated")
    if g and h:
        pooled = math.sqrt((g["std"] ** 2 + h["std"] ** 2) / 2.0)
        out["h_minus_g"] = float(h["mean"] - g["mean"])
        out["cohens_d_h_minus_g"] = float((h["mean"] - g["mean"]) / pooled) if pooled > 0 else None
        labels = [1 if int(row["label"]) == 0 else 0 for row in rows]
        scores = [float(row[metric]) for row in rows]
        out["auc_hallucinated_high"] = binary_auc(labels, scores)
    return out


def gt_target_indices(record):
    return [i for i, name in enumerate(record["target_names"]) if str(name).startswith("gt:")]


def node_target_index(record):
    for i, name in enumerate(record["target_names"]):
        if name == "node":
            return i
    for i, name in enumerate(record["target_names"]):
        if str(name).startswith("node_variant:"):
            return i
    return 0


def record_metrics(record, start_layer, end_layer):
    contrib = record["head_logit_contrib"].detach().cpu().numpy()
    band = contrib[start_layer: end_layer + 1]
    actual = band[:, :, 0]

    row = {
        "object_index": int(record["object_index"]),
        "image_id": int(record["image_id"]),
        "label": int(record["label"]),
        "surface_word": record["surface_word"],
        "node_word": record["node_word"],
        "token_step": int(record["token_step"]),
        "svar": float(record["svar"]),
        "actual_mean": float(actual.mean()),
        "actual_std": float(actual.std()),
        "actual_abs_mean": float(np.abs(actual).mean()),
        "actual_pos_sum": float(actual[actual > 0].sum()) if np.any(actual > 0) else 0.0,
        "actual_neg_sum": float(actual[actual < 0].sum()) if np.any(actual < 0) else 0.0,
        "actual_max": float(actual.max()),
        "actual_min": float(actual.min()),
        "actual_range": float(actual.max() - actual.min()),
    }

    flat_abs = np.sort(np.abs(actual.reshape(-1)))[::-1]
    total_abs = float(flat_abs.sum())
    row["actual_top10_abs_share"] = float(flat_abs[:10].sum() / total_abs) if total_abs > 0 else 0.0

    gt_idxs = gt_target_indices(record)
    if gt_idxs:
        node_idx = node_target_index(record)
        node_contrib = band[:, :, node_idx]
        gt_contrib = band[:, :, gt_idxs].max(axis=2)
        margin = node_contrib - gt_contrib
        row["node_minus_gt_max_mean"] = float(margin.mean())
        row["node_minus_gt_max_std"] = float(margin.std())
        row["node_minus_gt_max_pos_share"] = float((margin > 0).mean())
    else:
        row["node_minus_gt_max_mean"] = ""
        row["node_minus_gt_max_std"] = ""
        row["node_minus_gt_max_pos_share"] = ""

    return row


def write_csv(rows, path):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def per_head_stats(records, start_layer, end_layer):
    if not records:
        return []

    matrices = np.stack([
        record["head_logit_contrib"].detach().cpu().numpy()[:, :, 0]
        for record in records
    ])
    labels = np.array([int(record["label"]) for record in records])

    rows = []
    for layer_idx in range(start_layer, end_layer + 1):
        for head_idx in range(matrices.shape[2]):
            vals = matrices[:, layer_idx, head_idx]
            grounded = vals[labels == 1]
            hallucinated = vals[labels == 0]
            if len(grounded) == 0 or len(hallucinated) == 0:
                continue
            pooled = math.sqrt((grounded.std() ** 2 + hallucinated.std() ** 2) / 2.0)
            auc = binary_auc((labels == 0).astype(int), vals)
            rows.append({
                "layer": layer_idx,
                "head": head_idx,
                "grounded_mean": float(grounded.mean()),
                "hallucinated_mean": float(hallucinated.mean()),
                "h_minus_g": float(hallucinated.mean() - grounded.mean()),
                "cohens_d_h_minus_g": float((hallucinated.mean() - grounded.mean()) / pooled) if pooled > 0 else "",
                "auc_hallucinated_high": auc if auc is not None else "",
            })
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch_load_compat(args.contrib_file, map_location="cpu")
    records = payload["records"]

    rows = [record_metrics(record, args.start_layer, args.end_layer) for record in records]
    write_csv(rows, output_dir / "stage1_head_contrib_record_metrics.csv")

    metric_names = [
        "svar",
        "actual_mean",
        "actual_std",
        "actual_abs_mean",
        "actual_pos_sum",
        "actual_neg_sum",
        "actual_range",
        "actual_top10_abs_share",
        "node_minus_gt_max_mean",
        "node_minus_gt_max_pos_share",
    ]
    metric_summary = {}
    for metric in metric_names:
        valid_rows = [row for row in rows if row.get(metric) != ""]
        if valid_rows:
            metric_summary[metric] = describe_by_label(valid_rows, metric)

    head_rows = per_head_stats(records, args.start_layer, args.end_layer)
    write_csv(head_rows, output_dir / "stage1_per_head_actual_contrib_stats.csv")

    ranked_heads = sorted(
        head_rows,
        key=lambda row: abs(float(row["cohens_d_h_minus_g"])) if row["cohens_d_h_minus_g"] != "" else 0.0,
        reverse=True,
    )[: args.top_k]

    summary = {
        "contrib_file": args.contrib_file,
        "num_records": len(records),
        "num_grounded": int(sum(int(r["label"]) == 1 for r in records)),
        "num_hallucinated": int(sum(int(r["label"]) == 0 for r in records)),
        "layer_band": [args.start_layer, args.end_layer],
        "metric_summary": metric_summary,
        "top_heads_by_abs_effect": ranked_heads,
        "stage1_pass_candidates": [
            name for name, stats in metric_summary.items()
            if stats.get("auc_hallucinated_high") is not None
            and abs(float(stats["auc_hallucinated_high"]) - 0.5) >= 0.10
        ],
    }

    with open(output_dir / "stage1_head_contrib_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "num_records": summary["num_records"],
        "num_grounded": summary["num_grounded"],
        "num_hallucinated": summary["num_hallucinated"],
        "stage1_pass_candidates": summary["stage1_pass_candidates"],
        "top_heads_by_abs_effect": ranked_heads[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
