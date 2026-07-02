#!/usr/bin/env python
"""
Sanity checks for Stage D.1 selected heads.

This runs without additional model forward passes. It uses the existing
head-logit contribution trace and the D.1 wrong-head JSON to quantify:
  - activation frequency under the contribution gate
  - train/test preservation of selected-head effects
  - grounded harm risk proxies
  - aggregate selected-head contribution estimates
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from stage1_analyze_head_contrib import binary_auc, write_csv
from stage1_common import torch_load_compat


def parse_args():
    parser = argparse.ArgumentParser(description="D.1 selected-head activation sanity checks.")
    parser.add_argument("--contrib-file", type=str, default="stage1_outputs_n500/stage1_head_logit_contrib.pt")
    parser.add_argument("--heads-json", type=str, default="stage1_outputs_n500_d1_wrong_heads/stage1_d1_wrong_heads.json")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs_n500_d1_sanity")
    parser.add_argument("--head-key", type=str, default="wrong_heads")
    parser.add_argument("--train-frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def stable_train_flag(image_id, seed, train_frac):
    key = f"{int(image_id)}:{int(seed)}".encode("utf-8")
    value = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < train_frac


def split_name(record, train_frac, seed):
    return "train" if stable_train_flag(record["image_id"], seed, train_frac) else "test"


def labels_hallucinated(records):
    return np.array([1 if int(record["label"]) == 0 else 0 for record in records], dtype=np.int64)


def selected_heads(payload, key):
    heads = payload.get(key)
    if heads is None:
        raise KeyError(f"{key!r} not found in heads JSON.")
    out = []
    seen = set()
    for idx, item in enumerate(heads):
        layer = int(item["layer"])
        head = int(item["head"])
        if (layer, head) in seen:
            continue
        seen.add((layer, head))
        out.append({
            "selection_index": idx,
            "selection_rank": int(item.get("selection_rank", idx + 1)),
            "layer": layer,
            "head": head,
            "activation_threshold": float(item.get("activation_threshold", 0.0)),
            "selection_score": float(item.get("selection_score", item.get("h_minus_g", 0.0))),
            "train_h_minus_g": float(item.get("h_minus_g", 0.0)),
            "train_auc_hallucinated_high": (
                float(item["auc_hallucinated_high"])
                if item.get("auc_hallucinated_high") is not None
                else None
            ),
        })
    return out


def contrib_values(records, head):
    layer = int(head["layer"])
    head_idx = int(head["head"])
    return np.array([
        float(record["head_logit_contrib"][layer, head_idx, 0].item())
        for record in records
    ], dtype=np.float64)


def describe_values(vals):
    if len(vals) == 0:
        return {
            "mean": None,
            "std": None,
            "q50": None,
            "q90": None,
        }
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
    }


def per_head_rows(records, heads, split):
    labels = labels_hallucinated(records)
    rows = []
    if len(records) == 0 or len(np.unique(labels)) < 2:
        return rows
    for head in heads:
        vals = contrib_values(records, head)
        threshold = float(head["activation_threshold"])
        active = vals > threshold
        grounded = vals[labels == 0]
        hallucinated = vals[labels == 1]
        grounded_active = active[labels == 0]
        hallucinated_active = active[labels == 1]
        pooled = math.sqrt((grounded.std() ** 2 + hallucinated.std() ** 2) / 2.0)
        h_minus_g = float(hallucinated.mean() - grounded.mean())
        auc = binary_auc(labels, vals)
        rows.append({
            "split": split,
            "selection_rank": head["selection_rank"],
            "layer": head["layer"],
            "head": head["head"],
            "activation_threshold": threshold,
            "n": int(len(records)),
            "grounded_n": int((labels == 0).sum()),
            "hallucinated_n": int((labels == 1).sum()),
            "grounded_mean": float(grounded.mean()),
            "hallucinated_mean": float(hallucinated.mean()),
            "h_minus_g": h_minus_g,
            "cohens_d_h_minus_g": float(h_minus_g / pooled) if pooled > 0 else "",
            "auc_hallucinated_high": auc if auc is not None else "",
            "grounded_activation_rate": float(grounded_active.mean()) if len(grounded_active) else "",
            "hallucinated_activation_rate": float(hallucinated_active.mean()) if len(hallucinated_active) else "",
            "activation_rate_lift_h_minus_g": (
                float(hallucinated_active.mean() - grounded_active.mean())
                if len(grounded_active) and len(hallucinated_active)
                else ""
            ),
            "all_activation_rate": float(active.mean()),
            "threshold_minus_grounded_mean": float(threshold - grounded.mean()),
            "threshold_minus_hallucinated_mean": float(threshold - hallucinated.mean()),
            "grounded_abs_mean": float(np.abs(grounded).mean()),
            "hallucinated_abs_mean": float(np.abs(hallucinated).mean()),
            "train_selection_score": head["selection_score"],
            "train_h_minus_g_from_d1": head["train_h_minus_g"],
            "train_auc_from_d1": head["train_auc_hallucinated_high"],
        })
    return rows


def selected_matrix(records, heads):
    if not records or not heads:
        return np.zeros((len(records), len(heads)), dtype=np.float64)
    cols = [contrib_values(records, head) for head in heads]
    return np.stack(cols, axis=1)


def record_activation_rows(records, heads, split):
    labels = labels_hallucinated(records)
    vals = selected_matrix(records, heads)
    thresholds = np.array([float(head["activation_threshold"]) for head in heads], dtype=np.float64)
    active = vals > thresholds.reshape(1, -1) if len(heads) else np.zeros_like(vals, dtype=bool)
    rows = []
    for idx, record in enumerate(records):
        active_vals = vals[idx] * active[idx] if len(heads) else np.array([], dtype=np.float64)
        over_threshold = vals[idx] - thresholds if len(heads) else np.array([], dtype=np.float64)
        rows.append({
            "split": split,
            "object_index": int(record["object_index"]),
            "image_id": int(record["image_id"]),
            "label": int(record["label"]),
            "label_name": "hallucinated" if int(record["label"]) == 0 else "grounded",
            "surface_word": record.get("surface_word", ""),
            "node_word": record.get("node_word", ""),
            "token_step": int(record["token_step"]),
            "svar": float(record["svar"]),
            "selected_head_count": int(len(heads)),
            "active_head_count": int(active[idx].sum()) if len(heads) else 0,
            "any_active": int(active[idx].any()) if len(heads) else 0,
            "active_head_fraction": float(active[idx].mean()) if len(heads) else 0.0,
            "selected_contrib_sum": float(vals[idx].sum()) if len(heads) else 0.0,
            "selected_contrib_mean": float(vals[idx].mean()) if len(heads) else 0.0,
            "active_contrib_sum": float(active_vals.sum()) if len(heads) else 0.0,
            "max_contrib_minus_threshold": float(over_threshold.max()) if len(heads) else 0.0,
            "mean_contrib_minus_threshold": float(over_threshold.mean()) if len(heads) else 0.0,
        })
    return rows


def rows_for_split(records, train_frac, seed, split):
    if split == "all":
        return records
    return [record for record in records if split_name(record, train_frac, seed) == split]


def score_summary(record_rows, score):
    labels = np.array([1 if int(row["label"]) == 0 else 0 for row in record_rows], dtype=np.int64)
    scores = np.array([float(row[score]) for row in record_rows], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return None
    grounded = scores[labels == 0]
    hallucinated = scores[labels == 1]
    auc = binary_auc(labels, scores)
    return {
        "score": score,
        "grounded_mean": float(grounded.mean()),
        "hallucinated_mean": float(hallucinated.mean()),
        "h_minus_g": float(hallucinated.mean() - grounded.mean()),
        "auc_hallucinated_high": auc,
        "oriented_auc": max(auc, 1.0 - auc) if auc is not None else None,
    }


def split_summary(record_rows, head_rows, split):
    labels = np.array([1 if int(row["label"]) == 0 else 0 for row in record_rows], dtype=np.int64)
    out = {
        "split": split,
        "n": int(len(record_rows)),
        "grounded_n": int((labels == 0).sum()),
        "hallucinated_n": int((labels == 1).sum()),
        "scores": {},
        "head_activation": {},
    }
    for score in [
        "active_head_count",
        "any_active",
        "active_head_fraction",
        "selected_contrib_sum",
        "selected_contrib_mean",
        "active_contrib_sum",
        "max_contrib_minus_threshold",
        "mean_contrib_minus_threshold",
    ]:
        cur = score_summary(record_rows, score)
        if cur is not None:
            out["scores"][score] = cur

    numeric_head_rows = [
        row for row in head_rows
        if row.get("grounded_activation_rate") != "" and row.get("hallucinated_activation_rate") != ""
    ]
    if numeric_head_rows:
        g_rates = np.array([float(row["grounded_activation_rate"]) for row in numeric_head_rows], dtype=np.float64)
        h_rates = np.array([float(row["hallucinated_activation_rate"]) for row in numeric_head_rows], dtype=np.float64)
        lifts = h_rates - g_rates
        out["head_activation"] = {
            "num_heads": int(len(numeric_head_rows)),
            "mean_grounded_activation_rate": float(g_rates.mean()),
            "mean_hallucinated_activation_rate": float(h_rates.mean()),
            "mean_activation_lift_h_minus_g": float(lifts.mean()),
            "heads_hallucinated_activation_gt_grounded": int((lifts > 0).sum()),
            "heads_activation_lift_positive_fraction": float((lifts > 0).mean()),
            "mean_grounded_abs_contrib": float(np.mean([float(row["grounded_abs_mean"]) for row in numeric_head_rows])),
        }
    return out


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def train_test_validation(head_rows):
    train = {
        (int(row["layer"]), int(row["head"])): row
        for row in head_rows
        if row["split"] == "train"
    }
    test = {
        (int(row["layer"]), int(row["head"])): row
        for row in head_rows
        if row["split"] == "test"
    }
    keys = sorted(set(train) & set(test))
    if not keys:
        return {}
    train_h = np.array([float(train[key]["h_minus_g"]) for key in keys], dtype=np.float64)
    test_h = np.array([float(test[key]["h_minus_g"]) for key in keys], dtype=np.float64)
    train_lift = np.array([float(train[key]["activation_rate_lift_h_minus_g"]) for key in keys], dtype=np.float64)
    test_lift = np.array([float(test[key]["activation_rate_lift_h_minus_g"]) for key in keys], dtype=np.float64)
    test_positive = test_h > 0
    return {
        "num_heads": int(len(keys)),
        "train_test_h_minus_g_pearson": pearson(train_h, test_h),
        "train_test_activation_lift_pearson": pearson(train_lift, test_lift),
        "test_h_minus_g_mean": float(test_h.mean()),
        "test_h_minus_g_positive_fraction": float(test_positive.mean()),
        "test_activation_lift_mean": float(test_lift.mean()),
        "test_activation_lift_positive_fraction": float((test_lift > 0).mean()),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.heads_json) as f:
        heads_payload = json.load(f)
    config = heads_payload.get("config", {})
    train_frac = args.train_frac if args.train_frac is not None else float(config.get("train_frac", 0.5))
    seed = args.seed if args.seed is not None else int(config.get("seed", 927))

    contrib_payload = torch_load_compat(args.contrib_file, map_location="cpu")
    records = contrib_payload["records"]
    heads = selected_heads(heads_payload, args.head_key)

    all_head_rows = []
    all_record_rows = []
    split_summaries = {}
    for split in ["train", "test", "all"]:
        split_records = rows_for_split(records, train_frac, seed, split)
        head_rows = per_head_rows(split_records, heads, split)
        record_rows = record_activation_rows(split_records, heads, split)
        all_head_rows.extend(head_rows)
        all_record_rows.extend(record_rows)
        split_summaries[split] = split_summary(record_rows, head_rows, split)

    validation = train_test_validation(all_head_rows)

    summary = {
        "contrib_file": args.contrib_file,
        "heads_json": args.heads_json,
        "head_key": args.head_key,
        "num_records": len(records),
        "num_heads": len(heads),
        "train_frac": train_frac,
        "seed": seed,
        "split_summaries": split_summaries,
        "train_test_validation": validation,
        "quick_read": {
            "test_any_active_grounded_mean": split_summaries.get("test", {}).get("scores", {}).get("any_active", {}).get("grounded_mean"),
            "test_any_active_hallucinated_mean": split_summaries.get("test", {}).get("scores", {}).get("any_active", {}).get("hallucinated_mean"),
            "test_active_head_count_grounded_mean": split_summaries.get("test", {}).get("scores", {}).get("active_head_count", {}).get("grounded_mean"),
            "test_active_head_count_hallucinated_mean": split_summaries.get("test", {}).get("scores", {}).get("active_head_count", {}).get("hallucinated_mean"),
            "test_selected_sum_auc": split_summaries.get("test", {}).get("scores", {}).get("selected_contrib_sum", {}).get("auc_hallucinated_high"),
            "test_activation_lift_positive_fraction": validation.get("test_activation_lift_positive_fraction"),
            "test_h_minus_g_positive_fraction": validation.get("test_h_minus_g_positive_fraction"),
        },
    }

    write_csv(all_head_rows, output_dir / "stage1_d1_selected_head_activation_by_split.csv")
    write_csv(all_record_rows, output_dir / "stage1_d1_record_activation_summary.csv")
    write_csv(
        [
            {"split": split, "score": score, **stats}
            for split, split_stats in split_summaries.items()
            for score, stats in split_stats.get("scores", {}).items()
        ],
        output_dir / "stage1_d1_aggregate_score_summary.csv",
    )
    with open(output_dir / "stage1_d1_activation_sanity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "num_records": summary["num_records"],
        "num_heads": summary["num_heads"],
        "quick_read": summary["quick_read"],
        "train_test_validation": validation,
        "outputs": {
            "summary": str(output_dir / "stage1_d1_activation_sanity_summary.json"),
            "head_csv": str(output_dir / "stage1_d1_selected_head_activation_by_split.csv"),
            "record_csv": str(output_dir / "stage1_d1_record_activation_summary.csv"),
            "aggregate_csv": str(output_dir / "stage1_d1_aggregate_score_summary.csv"),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
