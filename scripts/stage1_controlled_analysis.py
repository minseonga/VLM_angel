#!/usr/bin/env python
"""
Controlled Stage 1 analysis for paper-level checks.

Inputs the D2 head-logit contribution trace and produces:
  - aggregate metric AUROC with bootstrap confidence intervals
  - token_step within-bucket analysis
  - SVAR-matched grounded/hallucinated pair analysis
  - logistic regression controls: label ~ metric + SVAR + token_step
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stage1_analyze_head_contrib import binary_auc, record_metrics, write_csv
from stage1_common import torch_load_compat


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
    parser = argparse.ArgumentParser(description="Stage 1 controlled analysis.")
    parser.add_argument("--contrib-file", type=str, default="stage1_outputs/stage1_head_logit_contrib.pt")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs")
    parser.add_argument("--start-layer", type=int, default=5)
    parser.add_argument("--end-layer", type=int, default=18)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=927)
    parser.add_argument("--token-step-bins", type=int, default=4)
    parser.add_argument("--min-bucket-class-count", type=int, default=5)
    parser.add_argument("--svar-match-threshold", type=float, default=0.05)
    parser.add_argument("--token-step-match-threshold", type=float, default=None)
    return parser.parse_args()


def as_float(value):
    if value == "" or value is None:
        return None
    return float(value)


def valid_metric_rows(rows, metric):
    return [row for row in rows if as_float(row.get(metric)) is not None]


def labels_hallucinated(rows):
    return np.array([1 if int(row["label"]) == 0 else 0 for row in rows], dtype=np.int64)


def metric_scores(rows, metric):
    return np.array([float(row[metric]) for row in rows], dtype=np.float64)


def bootstrap_auc_ci(labels, scores, iters, rng):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    auc = binary_auc(labels, scores)
    if auc is None:
        return {"auc": None, "ci95": None, "n_boot": 0}

    boot = []
    n = len(labels)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
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


def bootstrap_mean_ci(values, iters, rng):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": None, "ci95": None, "n_boot": 0}
    boot = []
    for _ in range(iters):
        idx = rng.integers(0, len(values), size=len(values))
        boot.append(float(values[idx].mean()))
    boot = np.array(boot, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n_boot": int(len(boot)),
    }


def cohen_d_h_minus_g(rows, metric):
    grounded = np.array([float(row[metric]) for row in rows if int(row["label"]) == 1], dtype=np.float64)
    hallucinated = np.array([float(row[metric]) for row in rows if int(row["label"]) == 0], dtype=np.float64)
    if len(grounded) == 0 or len(hallucinated) == 0:
        return None
    pooled = math.sqrt((grounded.std() ** 2 + hallucinated.std() ** 2) / 2.0)
    if pooled == 0:
        return None
    return float((hallucinated.mean() - grounded.mean()) / pooled)


def summarize_metric(rows, metric, bootstrap_iters, rng):
    cur_rows = valid_metric_rows(rows, metric)
    if not cur_rows:
        return None
    labels = labels_hallucinated(cur_rows)
    scores = metric_scores(cur_rows, metric)
    if len(np.unique(labels)) < 2:
        return None

    grounded = scores[labels == 0]
    hallucinated = scores[labels == 1]
    auc_info = bootstrap_auc_ci(labels, scores, bootstrap_iters, rng)
    oriented_auc = None
    if auc_info["auc"] is not None:
        oriented_auc = max(auc_info["auc"], 1.0 - auc_info["auc"])

    return {
        "metric": metric,
        "n": int(len(cur_rows)),
        "grounded_n": int(len(grounded)),
        "hallucinated_n": int(len(hallucinated)),
        "grounded_mean": float(grounded.mean()),
        "hallucinated_mean": float(hallucinated.mean()),
        "h_minus_g": float(hallucinated.mean() - grounded.mean()),
        "cohens_d_h_minus_g": cohen_d_h_minus_g(cur_rows, metric),
        "auc_hallucinated_high": auc_info["auc"],
        "auc_hallucinated_high_ci95": auc_info["ci95"],
        "oriented_auc": oriented_auc,
        "n_boot": auc_info["n_boot"],
    }


def raw_metric_summary(rows, metrics, bootstrap_iters, rng):
    out = {}
    for metric in metrics:
        summary = summarize_metric(rows, metric, bootstrap_iters, rng)
        if summary is not None:
            out[metric] = summary
    return out


def token_step_bucket_rows(rows, num_bins):
    steps = np.array([float(row["token_step"]) for row in rows], dtype=np.float64)
    if len(steps) == 0:
        return []
    quantiles = np.quantile(steps, np.linspace(0, 1, num_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    buckets = []
    for idx in range(num_bins):
        lo = quantiles[idx]
        hi = quantiles[idx + 1]
        bucket = [
            row for row in rows
            if float(row["token_step"]) > lo and float(row["token_step"]) <= hi
        ]
        buckets.append({
            "bucket": idx,
            "lo": None if np.isneginf(lo) else float(lo),
            "hi": None if np.isposinf(hi) else float(hi),
            "rows": bucket,
        })
    return buckets


def token_step_bucket_analysis(rows, metrics, num_bins, min_class_count, bootstrap_iters, rng):
    bucket_summaries = []
    aggregate = {}
    buckets = token_step_bucket_rows(rows, num_bins)

    for bucket in buckets:
        bucket_rows = bucket["rows"]
        g = sum(int(row["label"]) == 1 for row in bucket_rows)
        h = sum(int(row["label"]) == 0 for row in bucket_rows)
        entry = {
            "bucket": bucket["bucket"],
            "token_step_lo": bucket["lo"],
            "token_step_hi": bucket["hi"],
            "n": len(bucket_rows),
            "grounded_n": g,
            "hallucinated_n": h,
            "metrics": {},
        }
        if g >= min_class_count and h >= min_class_count:
            for metric in metrics:
                summary = summarize_metric(bucket_rows, metric, bootstrap_iters, rng)
                if summary is not None:
                    entry["metrics"][metric] = summary
        bucket_summaries.append(entry)

    for metric in metrics:
        vals = []
        for entry in bucket_summaries:
            summary = entry["metrics"].get(metric)
            if summary is None:
                continue
            vals.append(summary["auc_hallucinated_high"])
        vals = [v for v in vals if v is not None]
        if vals:
            aggregate[metric] = {
                "num_valid_buckets": len(vals),
                "mean_auc_hallucinated_high": float(np.mean(vals)),
                "mean_oriented_auc": float(np.mean([max(v, 1.0 - v) for v in vals])),
            }

    return {
        "buckets": bucket_summaries,
        "aggregate": aggregate,
    }


def svar_bin_analysis(rows, metrics, num_bins, min_class_count, bootstrap_iters, rng):
    svar = np.array([float(row["svar"]) for row in rows], dtype=np.float64)
    if len(svar) == 0:
        return {"bins": [], "aggregate": {}}
    quantiles = np.quantile(svar, np.linspace(0, 1, num_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    bins = []
    aggregate = {}
    for idx in range(num_bins):
        lo = quantiles[idx]
        hi = quantiles[idx + 1]
        bin_rows = [row for row in rows if float(row["svar"]) > lo and float(row["svar"]) <= hi]
        g = sum(int(row["label"]) == 1 for row in bin_rows)
        h = sum(int(row["label"]) == 0 for row in bin_rows)
        entry = {
            "bin": idx,
            "svar_lo": None if np.isneginf(lo) else float(lo),
            "svar_hi": None if np.isposinf(hi) else float(hi),
            "n": len(bin_rows),
            "grounded_n": g,
            "hallucinated_n": h,
            "metrics": {},
        }
        if g >= min_class_count and h >= min_class_count:
            for metric in metrics:
                summary = summarize_metric(bin_rows, metric, bootstrap_iters, rng)
                if summary is not None:
                    entry["metrics"][metric] = summary
        bins.append(entry)

    for metric in metrics:
        vals = []
        for entry in bins:
            summary = entry["metrics"].get(metric)
            if summary is None:
                continue
            vals.append(summary["auc_hallucinated_high"])
        vals = [v for v in vals if v is not None]
        if vals:
            aggregate[metric] = {
                "num_valid_bins": len(vals),
                "mean_auc_hallucinated_high": float(np.mean(vals)),
                "mean_oriented_auc": float(np.mean([max(v, 1.0 - v) for v in vals])),
            }

    return {
        "bins": bins,
        "aggregate": aggregate,
    }


def greedy_svar_matches(rows, svar_threshold, token_step_threshold=None):
    grounded = [row for row in rows if int(row["label"]) == 1]
    hallucinated = [row for row in rows if int(row["label"]) == 0]
    used_grounded = set()
    pairs = []

    hallucinated = sorted(hallucinated, key=lambda row: float(row["svar"]))
    for h_row in hallucinated:
        candidates = []
        for g_idx, g_row in enumerate(grounded):
            if g_idx in used_grounded:
                continue
            svar_diff = abs(float(h_row["svar"]) - float(g_row["svar"]))
            if svar_diff > svar_threshold:
                continue
            token_step_diff = abs(float(h_row["token_step"]) - float(g_row["token_step"]))
            if token_step_threshold is not None and token_step_diff > token_step_threshold:
                continue
            candidates.append((svar_diff, token_step_diff, g_idx, g_row))
        if not candidates:
            continue
        _, _, g_idx, g_row = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
        used_grounded.add(g_idx)
        pairs.append((h_row, g_row))
    return pairs


def matched_pair_analysis(rows, metrics, svar_threshold, token_step_threshold, bootstrap_iters, rng):
    pairs = greedy_svar_matches(rows, svar_threshold, token_step_threshold)
    out = {
        "svar_threshold": svar_threshold,
        "token_step_threshold": token_step_threshold,
        "num_pairs": len(pairs),
        "metrics": {},
        "pair_rows": [],
    }

    for h_row, g_row in pairs:
        out["pair_rows"].append({
            "hallucinated_object_index": h_row["object_index"],
            "grounded_object_index": g_row["object_index"],
            "hallucinated_image_id": h_row["image_id"],
            "grounded_image_id": g_row["image_id"],
            "hallucinated_node_word": h_row["node_word"],
            "grounded_node_word": g_row["node_word"],
            "hallucinated_svar": h_row["svar"],
            "grounded_svar": g_row["svar"],
            "svar_diff": float(h_row["svar"]) - float(g_row["svar"]),
            "hallucinated_token_step": h_row["token_step"],
            "grounded_token_step": g_row["token_step"],
            "token_step_diff": float(h_row["token_step"]) - float(g_row["token_step"]),
        })

    for metric in metrics:
        diffs = []
        valid_pair_count = 0
        for h_row, g_row in pairs:
            h_val = as_float(h_row.get(metric))
            g_val = as_float(g_row.get(metric))
            if h_val is None or g_val is None:
                continue
            diffs.append(h_val - g_val)
            valid_pair_count += 1
        if not diffs:
            continue
        diffs = np.array(diffs, dtype=np.float64)
        ci = bootstrap_mean_ci(diffs, bootstrap_iters, rng)
        out["metrics"][metric] = {
            "num_pairs": int(valid_pair_count),
            "mean_h_minus_g": ci["mean"],
            "mean_h_minus_g_ci95": ci["ci95"],
            "n_boot": ci["n_boot"],
            "positive_fraction": float((diffs > 0).mean()),
            "abs_mean": float(np.abs(diffs).mean()),
        }

    return out


def standardize_train_test(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def logistic_control_analysis(rows, metrics, bootstrap_iters, rng):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        import warnings
    except Exception as exc:
        return {"error": f"sklearn unavailable: {exc}"}

    out = {}
    for metric in metrics:
        cur_rows = valid_metric_rows(rows, metric)
        if len(cur_rows) < 10:
            continue
        y = labels_hallucinated(cur_rows)
        if len(np.unique(y)) < 2:
            continue
        X = np.array([
            [float(row[metric]), float(row["svar"]), float(row["token_step"])]
            for row in cur_rows
        ], dtype=np.float64)
        X = standardize_train_test(X)

        def fit_score(indices):
            x_i = X[indices]
            y_i = y[indices]
            if len(np.unique(y_i)) < 2:
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf = LogisticRegression(max_iter=1000, solver="liblinear")
                clf.fit(x_i, y_i)
            prob = clf.predict_proba(x_i)[:, 1]
            auc = binary_auc(y_i, prob)
            return {
                "metric_coef": float(clf.coef_[0][0]),
                "svar_coef": float(clf.coef_[0][1]),
                "token_step_coef": float(clf.coef_[0][2]),
                "auc_in_sample": auc,
            }

        base = fit_score(np.arange(len(y)))
        if base is None:
            continue

        boot_metric_coef = []
        boot_auc = []
        for _ in range(bootstrap_iters):
            idx = rng.integers(0, len(y), size=len(y))
            cur = fit_score(idx)
            if cur is None:
                continue
            boot_metric_coef.append(cur["metric_coef"])
            if cur["auc_in_sample"] is not None:
                boot_auc.append(cur["auc_in_sample"])

        out[metric] = {
            "n": int(len(y)),
            "grounded_n": int((y == 0).sum()),
            "hallucinated_n": int((y == 1).sum()),
            "metric_coef": base["metric_coef"],
            "svar_coef": base["svar_coef"],
            "token_step_coef": base["token_step_coef"],
            "auc_in_sample": base["auc_in_sample"],
            "metric_coef_ci95": [
                float(np.quantile(boot_metric_coef, 0.025)),
                float(np.quantile(boot_metric_coef, 0.975)),
            ] if boot_metric_coef else None,
            "auc_in_sample_ci95": [
                float(np.quantile(boot_auc, 0.025)),
                float(np.quantile(boot_auc, 0.975)),
            ] if boot_auc else None,
            "n_boot": int(len(boot_metric_coef)),
        }
    return out


def csv_safe_rows(rows):
    out = []
    for row in rows:
        out.append({k: v for k, v in row.items() if not isinstance(v, (dict, list, tuple))})
    return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch_load_compat(args.contrib_file, map_location="cpu")
    records = payload["records"]
    rows = [record_metrics(record, args.start_layer, args.end_layer) for record in records]
    write_csv(rows, output_dir / "stage1_controlled_record_metrics.csv")

    raw_summary = raw_metric_summary(rows, args.metrics, args.bootstrap_iters, rng)
    token_step = token_step_bucket_analysis(
        rows,
        args.metrics,
        args.token_step_bins,
        args.min_bucket_class_count,
        args.bootstrap_iters,
        rng,
    )
    svar_bins = svar_bin_analysis(
        rows,
        args.metrics,
        args.token_step_bins,
        args.min_bucket_class_count,
        args.bootstrap_iters,
        rng,
    )
    matched_svar = matched_pair_analysis(
        rows,
        args.metrics,
        args.svar_match_threshold,
        None,
        args.bootstrap_iters,
        rng,
    )
    matched_svar_token = None
    if args.token_step_match_threshold is not None:
        matched_svar_token = matched_pair_analysis(
            rows,
            args.metrics,
            args.svar_match_threshold,
            args.token_step_match_threshold,
            args.bootstrap_iters,
            rng,
        )

    logistic = logistic_control_analysis(rows, args.metrics, args.bootstrap_iters, rng)

    summary = {
        "contrib_file": args.contrib_file,
        "num_records": len(rows),
        "num_grounded": int(sum(int(row["label"]) == 1 for row in rows)),
        "num_hallucinated": int(sum(int(row["label"]) == 0 for row in rows)),
        "layer_band": [args.start_layer, args.end_layer],
        "metrics": args.metrics,
        "bootstrap_iters": args.bootstrap_iters,
        "raw_metric_summary": raw_summary,
        "token_step_bucket_analysis": token_step,
        "svar_bin_analysis": svar_bins,
        "svar_matched_analysis": matched_svar,
        "svar_token_step_matched_analysis": matched_svar_token,
        "logistic_control_analysis": logistic,
    }

    with open(output_dir / "stage1_controlled_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    raw_rows = [v for v in raw_summary.values()]
    write_csv(csv_safe_rows(raw_rows), output_dir / "stage1_controlled_raw_metric_summary.csv")

    match_rows = []
    for metric, stats in matched_svar["metrics"].items():
        match_rows.append({"match_type": "svar", "metric": metric, **stats})
    if matched_svar_token is not None:
        for metric, stats in matched_svar_token["metrics"].items():
            match_rows.append({"match_type": "svar_token_step", "metric": metric, **stats})
    write_csv(csv_safe_rows(match_rows), output_dir / "stage1_controlled_matched_metric_summary.csv")
    write_csv(matched_svar["pair_rows"], output_dir / "stage1_controlled_svar_matched_pairs.csv")
    if matched_svar_token is not None:
        write_csv(
            matched_svar_token["pair_rows"],
            output_dir / "stage1_controlled_svar_token_step_matched_pairs.csv",
        )

    print(json.dumps({
        "num_records": summary["num_records"],
        "num_grounded": summary["num_grounded"],
        "num_hallucinated": summary["num_hallucinated"],
        "raw_top_oriented_auc": sorted(
            [
                {
                    "metric": metric,
                    "oriented_auc": stats.get("oriented_auc"),
                    "auc_hallucinated_high": stats.get("auc_hallucinated_high"),
                    "ci95": stats.get("auc_hallucinated_high_ci95"),
                }
                for metric, stats in raw_summary.items()
                if stats.get("oriented_auc") is not None
            ],
            key=lambda item: item["oriented_auc"],
            reverse=True,
        )[:5],
        "svar_matched_pairs": matched_svar["num_pairs"],
        "svar_token_step_matched_pairs": None if matched_svar_token is None else matched_svar_token["num_pairs"],
        "outputs": {
            "summary": str(output_dir / "stage1_controlled_analysis_summary.json"),
            "raw_csv": str(output_dir / "stage1_controlled_raw_metric_summary.csv"),
            "matched_csv": str(output_dir / "stage1_controlled_matched_metric_summary.csv"),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
