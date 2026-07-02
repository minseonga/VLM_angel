#!/usr/bin/env python
"""
Step-conditioned head dynamics analysis for Stage 1C.

This script asks whether head-logit contribution effects are stable within
decoding-step bins, and exports train-split head candidates for causal tests.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from stage1_analyze_head_contrib import binary_auc, record_metrics, write_csv
from stage1_common import torch_load_compat
from stage1_controlled_analysis import (
    cohen_d_h_minus_g,
    summarize_metric,
    token_step_bucket_rows,
)


DEFAULT_METRICS = [
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


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze step-conditioned head contribution dynamics.")
    parser.add_argument("--contrib-file", type=str, default="stage1_outputs/stage1_head_logit_contrib.pt")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs_step_head_dynamics")
    parser.add_argument("--start-layer", type=int, default=5)
    parser.add_argument("--end-layer", type=int, default=18)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--token-step-bins", type=int, default=8)
    parser.add_argument("--min-bucket-class-count", type=int, default=5)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--top-heads-per-bin", type=int, default=20)
    parser.add_argument("--candidate-head-count", type=int, default=16)
    parser.add_argument("--selection-bin", choices=["last", "max_hallucination_rate"], default="last")
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=927)
    return parser.parse_args()


def stable_train_flag(image_id, seed, train_frac):
    key = f"{int(image_id)}:{int(seed)}".encode("utf-8")
    value = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < train_frac


def row_key(row):
    return (int(row["image_id"]), int(row["object_index"]))


def split_rows(rows, records, train_frac, seed):
    train_keys = {
        (int(record["image_id"]), int(record["object_index"]))
        for record in records
        if stable_train_flag(record["image_id"], seed, train_frac)
    }
    train = [row for row in rows if row_key(row) in train_keys]
    test = [row for row in rows if row_key(row) not in train_keys]
    return train, test, train_keys


def record_lookup(records):
    return {
        (int(record["image_id"]), int(record["object_index"])): record
        for record in records
    }


def labels_hallucinated_from_records(records):
    return np.array([1 if int(record["label"]) == 0 else 0 for record in records], dtype=np.int64)


def per_head_bin_stats(records, row_subset, record_by_key, start_layer, end_layer):
    subset_records = [record_by_key[row_key(row)] for row in row_subset if row_key(row) in record_by_key]
    if not subset_records:
        return []
    labels = labels_hallucinated_from_records(subset_records)
    if len(np.unique(labels)) < 2:
        return []

    matrices = np.stack([
        record["head_logit_contrib"].detach().cpu().numpy()[:, :, 0]
        for record in subset_records
    ])

    rows = []
    for layer_idx in range(start_layer, end_layer + 1):
        for head_idx in range(matrices.shape[2]):
            vals = matrices[:, layer_idx, head_idx]
            grounded = vals[labels == 0]
            hallucinated = vals[labels == 1]
            if len(grounded) == 0 or len(hallucinated) == 0:
                continue
            pooled = math.sqrt((grounded.std() ** 2 + hallucinated.std() ** 2) / 2.0)
            auc = binary_auc(labels, vals)
            h_minus_g = float(hallucinated.mean() - grounded.mean())
            rows.append({
                "layer": layer_idx,
                "head": head_idx,
                "grounded_mean": float(grounded.mean()),
                "hallucinated_mean": float(hallucinated.mean()),
                "h_minus_g": h_minus_g,
                "cohens_d_h_minus_g": float(h_minus_g / pooled) if pooled > 0 else "",
                "auc_hallucinated_high": auc if auc is not None else "",
                "oriented_auc": max(auc, 1.0 - auc) if auc is not None else "",
                "support_direction": "grounded_high" if h_minus_g < 0 else "hallucinated_high",
            })
    return rows


def bucket_label_counts(bucket_rows):
    grounded = int(sum(int(row["label"]) == 1 for row in bucket_rows))
    hallucinated = int(sum(int(row["label"]) == 0 for row in bucket_rows))
    total = grounded + hallucinated
    return grounded, hallucinated, total


def metric_bucket_rows(rows, metrics, buckets, min_class_count, bootstrap_iters, rng):
    out = []
    for bucket in buckets:
        bucket_rows = bucket["rows"]
        grounded, hallucinated, total = bucket_label_counts(bucket_rows)
        base = {
            "bucket": bucket["bucket"],
            "token_step_lo": bucket["lo"],
            "token_step_hi": bucket["hi"],
            "n": total,
            "grounded_n": grounded,
            "hallucinated_n": hallucinated,
            "hallucination_rate": float(hallucinated / total) if total else "",
        }
        if grounded < min_class_count or hallucinated < min_class_count:
            out.append({**base, "metric": "", "skip_reason": "insufficient_class_count"})
            continue
        for metric in metrics:
            summary = summarize_metric(bucket_rows, metric, bootstrap_iters, rng)
            if summary is None:
                continue
            ci = summary.pop("auc_hallucinated_high_ci95", None)
            out.append({
                **base,
                **summary,
                "auc_ci95_low": ci[0] if ci else "",
                "auc_ci95_high": ci[1] if ci else "",
            })
    return out


def head_bucket_rows(records, rows, buckets, record_by_key, start_layer, end_layer, min_class_count, top_k):
    out = []
    for bucket in buckets:
        bucket_rows = bucket["rows"]
        grounded, hallucinated, total = bucket_label_counts(bucket_rows)
        if grounded < min_class_count or hallucinated < min_class_count:
            continue
        stats = per_head_bin_stats(records, bucket_rows, record_by_key, start_layer, end_layer)
        for stat in stats:
            out.append({
                "bucket": bucket["bucket"],
                "token_step_lo": bucket["lo"],
                "token_step_hi": bucket["hi"],
                "n": total,
                "grounded_n": grounded,
                "hallucinated_n": hallucinated,
                "hallucination_rate": float(hallucinated / total) if total else "",
                **stat,
            })
    ranked = sorted(
        out,
        key=lambda row: abs(float(row["cohens_d_h_minus_g"])) if row["cohens_d_h_minus_g"] != "" else 0.0,
        reverse=True,
    )
    top = []
    counts = {}
    for row in ranked:
        bucket = row["bucket"]
        counts[bucket] = counts.get(bucket, 0)
        if counts[bucket] >= top_k:
            continue
        top.append(row)
        counts[bucket] += 1
    return out, top


def summarize_head_dynamics(head_rows):
    by_head = {}
    for row in head_rows:
        key = (int(row["layer"]), int(row["head"]))
        by_head.setdefault(key, []).append(row)

    out = []
    for (layer, head), rows in by_head.items():
        rows = sorted(rows, key=lambda row: int(row["bucket"]))
        effects = np.array([float(row["h_minus_g"]) for row in rows], dtype=np.float64)
        oriented = np.array([float(row["oriented_auc"]) for row in rows if row["oriented_auc"] != ""], dtype=np.float64)
        support_bins = int((effects < 0).sum())
        prior_bins = int((effects > 0).sum())
        out.append({
            "layer": layer,
            "head": head,
            "num_valid_bins": int(len(rows)),
            "mean_h_minus_g": float(effects.mean()),
            "std_h_minus_g": float(effects.std()),
            "first_h_minus_g": float(effects[0]),
            "last_h_minus_g": float(effects[-1]),
            "late_minus_early_h_minus_g": float(effects[-1] - effects[0]),
            "support_direction_bins": support_bins,
            "prior_direction_bins": prior_bins,
            "direction_consistency": float(max(support_bins, prior_bins) / len(rows)),
            "mean_oriented_auc": float(oriented.mean()) if len(oriented) else "",
            "max_oriented_auc": float(oriented.max()) if len(oriented) else "",
        })
    return sorted(
        out,
        key=lambda row: (
            float(row["direction_consistency"]),
            abs(float(row["late_minus_early_h_minus_g"])),
            float(row["mean_oriented_auc"]) if row["mean_oriented_auc"] != "" else 0.0,
        ),
        reverse=True,
    )


def select_bucket(buckets, rows):
    if not buckets:
        return None
    if not rows:
        return buckets[-1]["bucket"]
    rates = {}
    for bucket in buckets:
        bucket_rows = bucket["rows"]
        _, hallucinated, total = bucket_label_counts(bucket_rows)
        rates[bucket["bucket"]] = hallucinated / total if total else -1
    return max(rates, key=rates.get)


def head_entry(row):
    return {
        "layer": int(row["layer"]),
        "head": int(row["head"]),
        "bucket": int(row["bucket"]),
        "h_minus_g": float(row["h_minus_g"]),
        "cohens_d_h_minus_g": float(row["cohens_d_h_minus_g"]) if row["cohens_d_h_minus_g"] != "" else None,
        "auc_hallucinated_high": float(row["auc_hallucinated_high"]) if row["auc_hallucinated_high"] != "" else None,
        "oriented_auc": float(row["oriented_auc"]) if row["oriented_auc"] != "" else None,
        "support_direction": row["support_direction"],
    }


def select_candidate_heads(train_head_rows, all_head_rows, buckets, args):
    if args.selection_bin == "last":
        selection_bucket = buckets[-1]["bucket"] if buckets else None
    else:
        selection_bucket = select_bucket(buckets, train_head_rows)

    selected_bucket = None
    if selection_bucket is not None:
        for bucket in buckets:
            if int(bucket["bucket"]) == int(selection_bucket):
                selected_bucket = bucket
                break

    eligible = [
        row for row in train_head_rows
        if selection_bucket is not None and int(row["bucket"]) == int(selection_bucket)
    ]
    support = sorted(
        [row for row in eligible if float(row["h_minus_g"]) < 0],
        key=lambda row: (
            abs(float(row["cohens_d_h_minus_g"])) if row["cohens_d_h_minus_g"] != "" else 0.0,
            abs(float(row["h_minus_g"])),
        ),
        reverse=True,
    )[: args.candidate_head_count]
    prior = sorted(
        [row for row in eligible if float(row["h_minus_g"]) > 0],
        key=lambda row: (
            abs(float(row["cohens_d_h_minus_g"])) if row["cohens_d_h_minus_g"] != "" else 0.0,
            abs(float(row["h_minus_g"])),
        ),
        reverse=True,
    )[: args.candidate_head_count]

    all_heads = sorted({(int(row["layer"]), int(row["head"])) for row in all_head_rows})
    rng = random.Random(args.seed)
    rng.shuffle(all_heads)
    random_heads = [
        {"layer": layer, "head": head}
        for layer, head in all_heads[: args.candidate_head_count]
    ]

    return {
        "selection_bucket": int(selection_bucket) if selection_bucket is not None else None,
        "selection_token_step_lo": None if selected_bucket is None else selected_bucket["lo"],
        "selection_token_step_hi": None if selected_bucket is None else selected_bucket["hi"],
        "recommended_min_new_token_step": (
            0 if selected_bucket is None or selected_bucket["lo"] is None else int(math.floor(selected_bucket["lo"] + 1))
        ),
        "selection_bin_mode": args.selection_bin,
        "support_heads": [head_entry(row) for row in support],
        "prior_heads": [head_entry(row) for row in prior],
        "random_heads": random_heads,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch_load_compat(args.contrib_file, map_location="cpu")
    records = payload["records"]
    record_by_key = record_lookup(records)

    rows = [record_metrics(record, args.start_layer, args.end_layer) for record in records]
    train_rows, test_rows, train_keys = split_rows(rows, records, args.train_frac, args.seed)
    train_records = [record for record in records if (int(record["image_id"]), int(record["object_index"])) in train_keys]

    buckets = token_step_bucket_rows(rows, args.token_step_bins)
    metric_rows = metric_bucket_rows(
        rows,
        args.metrics,
        buckets,
        args.min_bucket_class_count,
        args.bootstrap_iters,
        rng,
    )

    all_head_rows, top_head_rows = head_bucket_rows(
        records,
        rows,
        buckets,
        record_by_key,
        args.start_layer,
        args.end_layer,
        args.min_bucket_class_count,
        args.top_heads_per_bin,
    )
    train_buckets = token_step_bucket_rows(train_rows, args.token_step_bins)
    train_head_rows, _ = head_bucket_rows(
        train_records,
        train_rows,
        train_buckets,
        record_by_key,
        args.start_layer,
        args.end_layer,
        args.min_bucket_class_count,
        args.top_heads_per_bin,
    )
    head_dynamics = summarize_head_dynamics(all_head_rows)
    candidates = select_candidate_heads(train_head_rows, all_head_rows, train_buckets, args)

    summary = {
        "contrib_file": args.contrib_file,
        "num_records": len(rows),
        "num_grounded": int(sum(int(row["label"]) == 1 for row in rows)),
        "num_hallucinated": int(sum(int(row["label"]) == 0 for row in rows)),
        "num_train_rows": len(train_rows),
        "num_test_rows": len(test_rows),
        "layer_band": [args.start_layer, args.end_layer],
        "token_step_bins": args.token_step_bins,
        "candidate_heads": candidates,
        "top_step_metric_rows": sorted(
            [
                row for row in metric_rows
                if row.get("metric") and row.get("oriented_auc") not in ("", None)
            ],
            key=lambda row: float(row["oriented_auc"]),
            reverse=True,
        )[:20],
        "top_step_heads": top_head_rows[:50],
        "top_head_dynamics": head_dynamics[:50],
    }

    write_csv(rows, output_dir / "stage1_step_head_record_metrics.csv")
    write_csv(metric_rows, output_dir / "stage1_step_bin_metric_summary.csv")
    write_csv(all_head_rows, output_dir / "stage1_step_bin_per_head_stats.csv")
    write_csv(top_head_rows, output_dir / "stage1_step_bin_top_heads.csv")
    write_csv(head_dynamics, output_dir / "stage1_step_head_dynamics_summary.csv")

    with open(output_dir / "stage1_causal_head_candidates.json", "w") as f:
        json.dump({
            "config": vars(args),
            "source": args.contrib_file,
            **candidates,
        }, f, indent=2)

    with open(output_dir / "stage1_step_head_dynamics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "num_records": summary["num_records"],
        "num_grounded": summary["num_grounded"],
        "num_hallucinated": summary["num_hallucinated"],
        "num_train_rows": summary["num_train_rows"],
        "num_test_rows": summary["num_test_rows"],
        "selection_bucket": candidates["selection_bucket"],
        "support_heads": candidates["support_heads"][:5],
        "prior_heads": candidates["prior_heads"][:5],
        "outputs": {
            "summary": str(output_dir / "stage1_step_head_dynamics_summary.json"),
            "metric_bins": str(output_dir / "stage1_step_bin_metric_summary.csv"),
            "per_head_bins": str(output_dir / "stage1_step_bin_per_head_stats.csv"),
            "candidates": str(output_dir / "stage1_causal_head_candidates.json"),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
